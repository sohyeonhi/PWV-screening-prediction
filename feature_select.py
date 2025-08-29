r"""
Nested CV + Stability Selection + 제한 임계값 탐색 (엄밀 모드, cutoff 기록/플래그 포함)
요구 반영:
  - 팔(A/B) 및 C on/off 결정을 **MLP 기준으로 통일** (각 outer fold에서 4조합 직접 학습/평가, 점수식: AUC − 0.5*gap − 0.5*overfit)
  - Test 평가 절차 교정: **train_pool K-fold CV 앙상블**(각 fold val로 early stopping)
  - Overfit cutoff **15%**: fold 레벨 과적합 판정식에서 사용
  - 추가: 단일 run 결과(decision.json)에 **cutoff 기록 및 pass/fail 플래그**(gap/overfit_rate) 저장
"""
from __future__ import annotations
import os, json, argparse, random
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import re, csv

# 프로젝트 모듈
import preprocess as pre
import train_model as tm
import mlp_eva as eva
try:
    import feature_step as fs
except Exception:
    fs = None

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score

# === Global cutoffs ===
GAP_CUTOFF = 0.15            # outer_mean_gap 기준(≤ 15% 통과)
OVERFIT_RATE_CUTOFF = 0.40   # outer_overfit_rate 기준(≤ 40% 통과)
AUC_DELTA_FOR_OVERFIT = 0.03 # AUC 과적합 보조 기준

# ---------------------------
# 블록 정의
# ---------------------------
A_BLOCK = ['T_CHOL','TG','HDL','LDL']
B_BLOCK = ['LDL/HDL','AIP','Non_HDL']
C_BLOCK = ['extreme_TG_flag','extreme_HDL_flag','extreme_LDL_flag','extreme_TCHOL_flag']
D_BLOCK = ['SBP','DBP']                           # 항상 포함
E_BLOCK = ['extreme_SBP_flag','extreme_DBP_flag'] # 그룹 토글
F_BLOCK = ['AST','ALT','RGTP']                    # 그룹 토글
MANDATORY_EXCLUDE = {'PWV'}                       # 입력 피처에서 제외(타깃은 PWV)

# ---------------------------
# 유틸
# ---------------------------

def build_feature_set(all_cols: List[str], arm: str, c_on: bool, base_toggle: Dict[str,bool]) -> List[str]:
    cols = [c for c in all_cols if c not in MANDATORY_EXCLUDE]
    include = set()
    if arm == 'A':
        include.update([c for c in A_BLOCK if c in cols])
    elif arm == 'B':
        include.update([c for c in B_BLOCK if c in cols])
    else:
        raise ValueError("arm must be 'A' or 'B'")
    if c_on:
        include.update([c for c in C_BLOCK if c in cols])
    include.update([c for c in D_BLOCK if c in cols])  # D 항상 포함
    if base_toggle.get('E', False):
        include.update([c for c in E_BLOCK if c in cols])
    if base_toggle.get('F', False):
        include.update([c for c in F_BLOCK if c in cols])
    block_union = set(A_BLOCK)|set(B_BLOCK)|set(C_BLOCK)|set(D_BLOCK)|set(E_BLOCK)|set(F_BLOCK)|MANDATORY_EXCLUDE
    remaining = [c for c in cols if c not in block_union]
    include.update(remaining)
    return sorted(include)


def prune_correlated(df: pd.DataFrame, feats: List[str], threshold: float=0.9) -> List[str]:
    if len(feats) <= 1:
        return feats
    X = df[feats].astype(float)
    try:
        corr = X.corr().abs()
    except Exception:
        return feats
    drop = set()
    for i in range(len(feats)):
        if feats[i] in drop:
            continue
        for j in range(i+1, len(feats)):
            if feats[j] in drop:
                continue
            try:
                if corr.iloc[i, j] > threshold:
                    drop.add(feats[j])
            except Exception:
                continue
    return [f for f in feats if f not in drop]

# ---------------------------
# 로더 & 평가
# ---------------------------

def make_loaders(df: pd.DataFrame, features: List[str], target: str,
                 train_idx: np.ndarray, val_idx: np.ndarray,
                 batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    X_tr = df.loc[train_idx, features].astype(float).values.astype(np.float32)
    y_tr = df.loc[train_idx, target].values.astype(np.float32)
    X_va = df.loc[val_idx, features].astype(float).values.astype(np.float32)
    y_va = df.loc[val_idx, target].values.astype(np.float32)
    ds_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    ds_va = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))
    return DataLoader(ds_tr, batch_size=batch_size, shuffle=True), DataLoader(ds_va, batch_size=batch_size, shuffle=False)


def eval_with_mlp(df: pd.DataFrame, features: List[str], target: str,
                  train_idx: np.ndarray, val_idx: np.ndarray,
                  threshold_grid: np.ndarray, threshold_mode: str='youden',
                  lr=1e-3, wd=1e-2, epochs=100, patience=8,
                  overfit_cutoff: float=GAP_CUTOFF, device=None) -> Dict[str, float]:
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    train_loader, val_loader = make_loaders(df, features, target, train_idx, val_idx)
    input_dim = len(features)
    model = tm.mlp_model(input_dim=input_dim, output_dim=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    model, train_losses, val_losses = tm.train_model(
        model=model, criterion=criterion, optimizer=optimizer,
        train_loader=train_loader, val_loader=val_loader,
        device=device, num_epochs=epochs, patience=patience, min_delta=1e-4
    )

    # Validation 예측
    _, _, val_auc, y_true_val, y_pred_val_fixed, y_prob_val = eva.evaluate_model(model, val_loader, criterion, device, plot=False)

    # Train 예측(갭/오버핏 계산용)
    model.eval()
    y_true_tr, y_prob_tr = [], []
    with torch.no_grad():
        for xb, yb in train_loader:
            xb = xb.to(device)
            prob = torch.sigmoid(model(xb)).cpu().numpy().ravel()
            y_true_tr.extend(yb.numpy().ravel().tolist())
            y_prob_tr.extend(prob.tolist())

    # 제한 임계값 탐색
    if threshold_mode == 'youden':
        best_th, best_score = 0.5, -1
        for th in threshold_grid:
            yp = (np.array(y_prob_val) >= th).astype(int)
            tp = ((yp==1)&(np.array(y_true_val)==1)).sum(); fp=((yp==1)&(np.array(y_true_val)==0)).sum()
            fn = ((yp==0)&(np.array(y_true_val)==1)).sum(); tn=((yp==0)&(np.array(y_true_val)==0)).sum()
            tpr = tp / max(1,(tp+fn)); fpr = fp / max(1,(fp+tn))
            j = tpr - fpr
            if j > best_score:
                best_score, best_th = j, float(th)
        best_f1 = f1_score(y_true_val, (np.array(y_prob_val) >= best_th).astype(int))
    else:
        best_th, best_f1 = 0.5, -1
        for th in threshold_grid:
            f1 = f1_score(y_true_val, (np.array(y_prob_val) >= th).astype(int))
            if f1 > best_f1:
                best_f1, best_th = float(f1), float(th)

    f1_fixed = f1_score(y_true_val, (np.array(y_prob_val) >= 0.5).astype(int))

    # gap & overfit 판정(0.5 기준 F1)
    f1_tr = f1_score(y_true_tr, (np.array(y_prob_tr) >= 0.5).astype(int))
    gap = abs(f1_tr - f1_fixed)
    overfit_flag = (f1_tr - f1_fixed > overfit_cutoff) or \
                   (roc_auc_score(y_true_tr, y_prob_tr) - val_auc > AUC_DELTA_FOR_OVERFIT)

    return {
        'val_auc': float(val_auc),
        'val_f1_fixed': float(f1_fixed),
        'val_f1_best': float(best_f1),
        'best_threshold': float(best_th),
        'gap': float(gap),
        'overfit': int(overfit_flag),
    }

# ---------------------------
# Stability Selection (Elastic Net)
# ---------------------------

def stability_selection(df: pd.DataFrame, candidate_feats: List[str], target: str,
                        tr_idx: np.ndarray, B: int = 100, subsample_rate: float=0.5,
                        C_val: float=0.3, l1_ratio: float=0.5) -> Tuple[pd.Series, Dict[str,float]]:
    rng = random.Random(2024)
    X_all = df.loc[tr_idx, candidate_feats].astype(float).values
    y_all = df.loc[tr_idx, target].values
    n = len(tr_idx)
    counts = pd.Series(0.0, index=candidate_feats, dtype=float)
    for b in range(B):
        m = max(10, int(n * subsample_rate))
        idx = np.array(rng.sample(range(n), m))
        X = X_all[idx]; y = y_all[idx]
        scaler = StandardScaler(); Xs = scaler.fit_transform(X)
        clf = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=l1_ratio,
                                 C=C_val, max_iter=2000, n_jobs=-1, random_state=rng.randrange(10**6))
        try:
            clf.fit(Xs, y)
            coef = getattr(clf, 'coef_', None)
            if coef is None:
                continue
            selected = (np.abs(coef).ravel() > 1e-8)
            counts[selected] += 1
        except Exception:
            continue
    freq = counts / max(1,B)
    def group_freq(group_list: List[str]) -> float:
        lst = [f for f in group_list if f in freq.index]
        if not lst:
            return 0.0
        return float((freq.loc[lst] > 0).mean())
    group = {'E_group': group_freq(E_BLOCK), 'F_group': group_freq(F_BLOCK)}
    return freq, group

# ---------------------------
# 메인 파이프라인
# ---------------------------

def run(df: pd.DataFrame, target: str, out_dir: str,
        outer_k: int = 5, stability_B: int = 100, stability_tau: float = 0.6,
        threshold_min: float=0.45, threshold_max: float=0.55, threshold_steps: int=11, threshold_mode: str='youden',
        subsample_rate: float=0.5, enet_C: float=0.3, enet_l1: float=0.5,
        corr_prune: float=0.9, seed: int=42,
        overfit_cutoff: float=GAP_CUTOFF, test_cv_k: int=5) -> Dict[str,object]:

    os.makedirs(out_dir, exist_ok=True)

    # 고정 테스트 인덱스
    fixed_test_idx = None
    if fs is not None and hasattr(fs, 'load_or_create_test_idx'):
        fixed_test_idx = fs.load_or_create_test_idx(df, target)

    all_cols = [c for c in df.columns if c != target]
    idx_all = np.arange(len(df))
    if fixed_test_idx is not None:
        mask = np.ones(len(df), dtype=bool); mask[fixed_test_idx] = False
        train_pool_idx = idx_all[mask]
        test_idx = np.array(fixed_test_idx)
    else:
        train_pool_idx = idx_all
        test_idx = np.array([], dtype=int)

    y_pool = df.loc[train_pool_idx, target].values
    outer = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=seed)

    chosen_arms, chosen_c = [], []
    rows_outer, fold_sets = [], []
    threshold_grid = np.linspace(threshold_min, threshold_max, threshold_steps)

    # ===== Step1: 팔/C 결정을 MLP로 =====
    for fold_id, (tr_rel, va_rel) in enumerate(outer.split(train_pool_idx, y_pool), start=1):
        tr_idx = train_pool_idx[tr_rel]; va_idx = train_pool_idx[va_rel]
        base_toggle = {'E': True, 'F': True}
        armc_scores: Dict[Tuple[str,bool], Dict[str,float]] = {}
        for arm in ['A','B']:
            for c_on in [False, True]:
                feats = build_feature_set(all_cols, arm=arm, c_on=c_on, base_toggle=base_toggle)
                s = eval_with_mlp(df, feats, target, tr_idx, va_idx, threshold_grid, threshold_mode=threshold_mode,
                                  lr=1e-3, wd=1e-2, epochs=100, patience=8, overfit_cutoff=overfit_cutoff)
                score = s['val_auc'] - 0.5*s['gap'] - 0.5*int(s['overfit'])
                armc_scores[(arm,c_on)] = {'score': float(score), **s}
        (arm_best, c_best) = max(armc_scores.keys(), key=lambda k: armc_scores[k]['score'])
        chosen_arms.append(arm_best); chosen_c.append(c_best)

        # ===== Step2: Stability Selection + 그룹 반영 + 높은 상관관계 제거 =====
        candidate = build_feature_set(all_cols, arm_best, c_best, base_toggle={'E': False, 'F': False})
        always = set(D_BLOCK) | (set(A_BLOCK) if arm_best=='A' else set(B_BLOCK)) | (set(C_BLOCK) if c_best else set())
        candidate = sorted(set(candidate) | always)
        freq, group = stability_selection(df, candidate, target, tr_idx,
                                          B=stability_B, subsample_rate=subsample_rate,
                                          C_val=enet_C, l1_ratio=enet_l1)
        selected = set(freq[freq >= stability_tau].index.tolist()) | always
        if group.get('E_group', 0.0) >= stability_tau:
            selected.update([f for f in E_BLOCK if f in candidate])
        if group.get('F_group', 0.0) >= stability_tau:
            selected.update([f for f in F_BLOCK if f in candidate])
        selected = prune_correlated(df, sorted(selected), threshold=corr_prune)

        # ===== Step3: MLP 학습 + 제한 임계값 탐색 =====
        res = eval_with_mlp(df, selected, target, tr_idx, va_idx, threshold_grid, threshold_mode=threshold_mode,
                            lr=1e-3, wd=1e-2, epochs=100, patience=8, overfit_cutoff=overfit_cutoff)

        rows_outer.append({
            'fold': fold_id,
            'arm': arm_best,
            'c_on': bool(c_best),
            'val_auc': res['val_auc'],
            'val_f1_fixed': res['val_f1_fixed'],
            'val_f1_best': res['val_f1_best'],
            'best_threshold': res['best_threshold'],
            'gap': res['gap'],
            'overfit': int(res['overfit']),
            'n_features': len(selected)
        })
        fold_sets.append({'fold': fold_id, 'features': '|'.join(selected)})

    # ===== 집계 =====
    def majority(xs):
        return max(set(xs), key=xs.count) if xs else None
    arm_final = majority(chosen_arms); c_final = majority(chosen_c)

    df_outer = pd.DataFrame(rows_outer)
    df_outer.to_csv(os.path.join(out_dir, 'outer_metrics0.3.csv'), index=False)
    pd.DataFrame(fold_sets).to_csv(os.path.join(out_dir, 'fold_feature_sets0.3.csv'), index=False)

    # 자카드
    sets = [set(s['features'].split('|')) for s in fold_sets]
    def jaccard(a: set, b: set) -> float:
        if not a and not b:
            return 1.0
        return float(len(a & b) / max(1, len(a | b)))
    J = np.zeros((len(sets), len(sets)))
    for i in range(len(sets)):
        for j in range(len(sets)):
            J[i,j] = jaccard(sets[i], sets[j])
    tri = J[np.triu_indices(len(sets), k=1)] if len(sets)>1 else np.array([1.0])
    jaccard_mean = float(tri.mean()) if tri.size else 1.0
    pd.DataFrame(J, columns=[f'fold{ i+1 }' for i in range(len(sets))], index=[f'fold{ i+1 }' for i in range(len(sets))]).to_csv(os.path.join(out_dir, 'jaccard_matrix0.3.csv'))

    # 전체 빈도
    all_selected = []
    for s in sets:
        all_selected += list(s)
    freq_overall = (pd.Series(all_selected).value_counts() / max(1,len(sets))).sort_values(ascending=False)
    df_stab = pd.DataFrame({'feature': freq_overall.index, 'frequency': freq_overall.values})
    df_stab.to_csv(os.path.join(out_dir, 'stability_feature_frequency0.3.csv'), index=False)

    # 최종 임계값: 중앙값
    best_th_final = float(np.median(df_outer['best_threshold'].values)) if not df_outer.empty else 0.5

    # ===== Test 평가 교정: CV 앙상블 =====
    test_metrics = None
    if 'test_idx' in locals() and test_idx.size > 0:
        always_final = set(D_BLOCK) | (set(A_BLOCK) if arm_final=='A' else set(B_BLOCK)) | (set(C_BLOCK) if c_final else set())
        final_feats = sorted(set(df_stab.loc[df_stab['frequency']>=stability_tau, 'feature'].tolist()) | always_final)
        final_feats = prune_correlated(df, final_feats, threshold=corr_prune)

        skf = StratifiedKFold(n_splits=test_cv_k, shuffle=True, random_state=seed+777)
        y_te_all, p_te_list = None, []
        val_f1s, val_aucs = [], []
        for tr_rel, va_rel in skf.split(train_pool_idx, y_pool):
            tr2 = train_pool_idx[tr_rel]; va2 = train_pool_idx[va_rel]
            res = eval_with_mlp(df, final_feats, target, tr2, va2, threshold_grid, threshold_mode=threshold_mode,
                                lr=1e-3, wd=1e-2, epochs=100, patience=8, overfit_cutoff=overfit_cutoff)
            val_f1s.append(res['val_f1_fixed'])
            val_aucs.append(res['val_auc'])
            X_te = df.loc[test_idx, final_feats].astype(float).values.astype(np.float32)
            y_te = df.loc[test_idx, target].values.astype(np.float32)
            y_te_all = y_te
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            train_loader, val_loader = make_loaders(df, final_feats, target, tr2, va2)
            model = tm.mlp_model(input_dim=len(final_feats), output_dim=1).to(device)
            criterion = nn.BCEWithLogitsLoss(); optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
            model, _, _ = tm.train_model(model, criterion, optimizer, train_loader, val_loader, device=device, num_epochs=100, patience=8, min_delta=1e-4)
            model.eval()
            with torch.no_grad():
                p_te = torch.sigmoid(model(torch.from_numpy(X_te).to(device))).cpu().numpy().ravel()
            p_te_list.append(p_te)
        p_te_mean = np.mean(np.vstack(p_te_list), axis=0)
        test_auc = float(roc_auc_score(y_te_all, p_te_mean))
        test_f1  = float(f1_score(y_te_all, (p_te_mean >= best_th_final).astype(int)))
        cv_auc   = float(np.mean(val_aucs))
        cv_f1    = float(np.mean(val_f1s))
        test_gap = float(abs(cv_f1 - test_f1))
        test_overfit_flag = int((cv_f1 - test_f1 > GAP_CUTOFF) or (cv_auc - test_auc > AUC_DELTA_FOR_OVERFIT))
        test_metrics = {'test_auc': test_auc, 'test_f1': test_f1, 'test_gap': test_gap, 'test_overfit_flag': test_overfit_flag, 'n_final_features': len(final_feats)}
        pd.DataFrame({'final_features':["|".join(final_feats)]}).to_csv(os.path.join(out_dir, 'final_features0.3.csv'), index=False)

    # ===== decision dict =====
    decision = {
        'outer_k': outer_k,
        'stability_B': stability_B,
        'stability_tau': stability_tau,
        'threshold_range': [float(threshold_min), float(threshold_max)],
        'threshold_mode': threshold_mode,
        'recommended_arm': arm_final,
        'recommended_C': 'ON' if c_final else 'OFF',
        'final_threshold': best_th_final,
        'outer_mean_auc': float(df_outer['val_auc'].mean()) if not df_outer.empty else None,
        'outer_mean_f1_fixed': float(df_outer['val_f1_fixed'].mean()) if not df_outer.empty else None,
        'outer_mean_f1_best': float(df_outer['val_f1_best'].mean()) if not df_outer.empty else None,
        'outer_mean_gap': float(df_outer['gap'].mean()) if not df_outer.empty else None,
        'outer_overfit_rate': float(df_outer['overfit'].mean()) if not df_outer.empty else None,
        'jaccard_mean': float(jaccard_mean),
    }
    if test_metrics is not None:
        decision.update(test_metrics)

    # record cutoffs & pass/fail flags for single-run visibility
    decision['gap_cutoff'] = float(GAP_CUTOFF)
    decision['overfit_rate_cutoff'] = float(OVERFIT_RATE_CUTOFF)
    decision['filter_pass_gap'] = int((decision.get('outer_mean_gap') or 0.0) <= GAP_CUTOFF)
    decision['filter_pass_overfit_rate'] = int((decision.get('outer_overfit_rate') or 0.0) <= OVERFIT_RATE_CUTOFF)

    with open(os.path.join(out_dir, 'decision.json'), 'w', encoding='utf-8') as f:
        json.dump(decision, f, ensure_ascii=False, indent=2)
    return decision

# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description='Nested CV (MLP arm scoring) + Test CV ensemble + Overfit cutoff (patched)')
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--target_column', type=str, default='PWV')
    ap.add_argument('--out_dir', type=str, required=True)

    ap.add_argument('--outer_k', type=int, default=5)
    ap.add_argument('--stability_B', type=int, default=100)
    ap.add_argument('--stability_tau', type=float, default=0.6)
    ap.add_argument('--subsample_rate', type=float, default=0.5)
    ap.add_argument('--enet_C', type=float, default=0.3)
    ap.add_argument('--enet_l1', type=float, default=0.5)
    ap.add_argument('--corr_prune', type=float, default=0.9)

    ap.add_argument('--threshold_min', type=float, default=0.45)
    ap.add_argument('--threshold_max', type=float, default=0.55)
    ap.add_argument('--threshold_steps', type=int, default=11)
    ap.add_argument('--threshold_mode', type=str, choices=['f1max','youden'], default='youden')

    ap.add_argument('--overfit_cutoff', type=float, default=GAP_CUTOFF)
    ap.add_argument('--test_cv_k', type=int, default=5)

    ap.add_argument('--seed', type=int, default=42)

    args = ap.parse_args()

    raw = pre.load_data(args.data_path)
    df = pre.preprocess_raw_data(raw)

    if args.target_column not in df.columns:
        raise KeyError(f"target_column='{args.target_column}' 이 데이터에 없습니다. 예시 열들: {list(df.columns)[:20]} ...")

    decision = run(df, target=args.target_column, out_dir=args.out_dir,
                   outer_k=args.outer_k, stability_B=args.stability_B, stability_tau=args.stability_tau,
                   threshold_min=args.threshold_min, threshold_max=args.threshold_max, threshold_steps=args.threshold_steps, threshold_mode=args.threshold_mode,
                   subsample_rate=args.subsample_rate, enet_C=args.enet_C, enet_l1=args.enet_l1,
                   corr_prune=args.corr_prune, seed=args.seed,
                   overfit_cutoff=args.overfit_cutoff, test_cv_k=args.test_cv_k)

    print(json.dumps(decision, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
