import os
import sys
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype, is_bool_dtype

# ===== 듀얼 임포트: 패키지 실행 우선, ▶ 실패 시 단일 실행 =====
try:
    from . import preprocess as pre
    # [CHANGED] basemodel을 우선 사용(동일 API)
    from . import basemodel as mlp  
    from . import mlp_eva as eva
    from . import feature_store as sf 
except ImportError:
    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    if CUR_DIR not in sys.path:
        sys.path.insert(0, CUR_DIR)
    import preprocess as pre
    import basemodel as mlp  
    import mlp_eva as eva
    import feature_store as sf

# ===== 비교 목록 =====
#  원시 그대로

Case_1 = ['AGE', 'SEX', 'SBP', 'DBP', 
          'AST', 'ALT', 'RGTP', "T_CHOL", "HDL", "LDL", "TG"]
Case_2 = ['AGE', 'SEX', 'SBP', 'DBP', 
          'AST', 'ALT', 'RGTP', 'LDL/HDL', 'AIP', 'Non_HDL']
Case_3 = ['AGE', 'SEX', 'SBP', 'DBP', 
          'AST', 'ALT', 'RGTP', 'LDL/HDL', 'AIP', 'Non_HDL','PAST_HISTORY_SCORE']    
Case_4 = ['AGE', 'SEX', 'SBP', 'DBP', 
          'AST', 'ALT', 'RGTP', "T_CHOL", "HDL", "LDL", "TG",'PAST_HISTORY_SCORE']          

STEPS = {
    "Case_1": Case_1,
    "Case_2": Case_2,
    "Case_3": Case_3,
    "Case_4": Case_4
} 

TARGET = "PWV"  # 타깃(라벨)

# =================== 시드/기준 상수 =====================
SEEDS = [17, 42, 48, 129, 252]          # 5개 시드 
GAP_CUTOFF = 0.15               # |gap| 15% 컷(절대값 기준)
AUC_MIN, F1_MIN = 0.75, 0.72    # 베이스라인 하한
# ================================================================

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15   # 전체 대비
TEST_RATIO  = 0.15   # 전체 대비
VAL_REL = VAL_RATIO / (1.0 - TEST_RATIO)  # 0.1764705882...

# 아티팩트 저장 폴더
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(SRC_DIR, "artifacts")
os.makedirs(ART_DIR, exist_ok=True)
TEST_IDX_PATH = os.path.join(ART_DIR, "fixed_test_idx.npy")

def make_holdout_indices(df: pd.DataFrame, target: str, test_size: float = TEST_RATIO, random_state: int = 42):
    y = df[target].values
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    (train_idx, test_idx), = splitter.split(df, y)
    return test_idx

def load_or_create_test_idx(df: pd.DataFrame, target: str) -> np.ndarray:
    if os.path.exists(TEST_IDX_PATH):
        test_idx = np.load(TEST_IDX_PATH)
        # 간단 검증: 길이/범위 체크 (필요시 강화 가능)
        assert test_idx.ndim == 1
        assert test_idx.max() < len(df) and test_idx.min() >= 0
        return test_idx
    # 없으면 생성해서 저장
    test_idx = make_holdout_indices(df, target, test_size=TEST_RATIO, random_state=42)
    np.save(TEST_IDX_PATH, test_idx)
    return test_idx

# 시드 고정 유틸
import random
def set_all_seeds(sd: int) -> None:
    random.seed(sd)
    np.random.seed(sd)
    try:
        import torch
        torch.manual_seed(sd)
        torch.cuda.manual_seed_all(sd)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _train_eval_once(df: pd.DataFrame, target: str, device,
                     lr=1e-3, batch_size=32, epochs=30, patience=2,
                     check_overfit: bool = True, plot_loss: bool = False,
                     fixed_test_idx=None
                     ) -> Tuple[float, float, Dict]:
    """
    df: (피처 + target)만 포함한 데이터프레임
    반환: (AUC, F1, extras) — extras에는 과적합 신호/갭 등 기록
    """
    # mlp.set_seed(42)
    # data split/scaling
    train_loader, val_loader, test_loader, input_dim = mlp.data_split(
        df, target_column=target, batch_size=batch_size,
        test_size=TEST_RATIO,
        val_size=VAL_REL,
        random_state=42,
        fixed_test_idx=fixed_test_idx if fixed_test_idx is not None else None
    )

    import torch
    model = mlp.mlp_model(input_dim, output_dim=2).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train
    model, train_losses, val_losses = mlp.train_model(
        model=model, 
        criterion=criterion, 
        optimizer=optimizer,
        train_loader=train_loader, 
        val_loader=val_loader, 
        device=device,
        num_epochs=epochs, 
        patience=patience, 
        scheduler=None
    )

    # evaluate
    _, _, auc, y_true, y_pred, _ = eva.evaluate_model(model, test_loader, criterion, device, plot=False)
    f1 = f1_score(y_true, y_pred, average="binary")

    extras: Dict = {}
    if check_overfit:
        extras = mlp.evaluate_overfitting(train_losses, val_losses, window=5, gap_threshold=GAP_CUTOFF)

    return float(auc), float(f1), extras

def run_steps_raw_only(df: pd.DataFrame, steps: Dict[str, List[str]], target: str, device,
                       lr=1e-3, batch_size=32, epochs=30, patience=2,
                       cumulative: bool = False, print_each: bool = True,
                       check_overfit: bool = True, plot_loss_each: bool = False,
                       fixed_test_idx=None):
    """
    returns: records = [{"label": "Comp 1", "auc":..., "f1":..., "overfit":..., "gap":...}, ...]
    """
    # mlp.set_seed(42)
    df0 = df.copy()
    records = []
    
    for label, keep_list in steps.items():
        plan_keep = list(dict.fromkeys(keep_list))
        exist    = [c for c in plan_keep if c in df0.columns]
        use_cols = exist + ([target] if target in df0.columns else [])

        auc_i, f1_i, extra = _train_eval_once(
            df0[use_cols].copy(), target, device,
            lr=lr, batch_size=batch_size, epochs=epochs, patience=patience,
            check_overfit=check_overfit, plot_loss=plot_loss_each,
            fixed_test_idx=fixed_test_idx
        )

        rec = {
            "label": label,
            "auc": auc_i,
            "f1": f1_i,
            "overfit": bool(extra.get("overfit", False)),
            "gap": float(extra.get("gap", 0.0)) if extra.get("gap") is not None else 0.0
        }
        records.append(rec)

        if print_each:
            of = "YES" if rec["overfit"] else "no"
            gap_pct = rec["gap"] * 100.0
            print(f"{label}: AUC={auc_i:.4f}, F1={f1_i:.4f} | overfit={of}, gap={gap_pct:.1f}%")

    return records

# ===================== multi-seed =====================
def run_multi_seed_eval(df: pd.DataFrame, steps: Dict[str,List[str]], target: str, device,
                        seeds: List[int] = SEEDS, lr=1e-3, batch_size=32, epochs=100, patience=8,
                        fixed_test_idx=None
                        ) -> Dict[str, List[Dict]]:
    """
    각 시드에 대해 run_steps_raw_only 실행하고, Comp별로 시드 결과 리스트를 모아 반환.
    return: {"Comp 1": [{"auc":..., "f1":..., "gap":..., "overfit":...}, ...], ...}
    """
    from collections import defaultdict
    comp2rows: Dict[str, List[Dict]] = defaultdict(list)

    for sd in seeds:
        set_all_seeds(sd)
        recs = run_steps_raw_only(
            df, steps, target, device,
            lr=lr, batch_size=batch_size, epochs=epochs, patience=patience,
            cumulative=False, print_each=False,
            check_overfit=True, plot_loss_each=False,
            fixed_test_idx = fixed_test_idx
        )
        for r in recs:
            comp2rows[r["label"]].append({
                "auc": r["auc"],
                "f1": r["f1"],
                "gap": r["gap"],             
                "overfit": r["overfit"]
            })
    return comp2rows

def print_comp_scores_per_seed(comp2rows: Dict[str, List[Dict]]) -> None:
    """
    시드별 결과 전부 출력:
    Comp n: AUC=..., F1=... | overfit=YES/no, gap=±x.x%
    (Comp별로 시드 수만큼 줄이 반복 출력됨)
    """

    for label in sorted(comp2rows.keys(), key=_label_numeric_key):
        rows = comp2rows[label]
        for r in rows:
            of = "YES" if r["overfit"] else "no"
            gap_pct = r["gap"] * 100.0
            print(f"{label}: AUC={r['auc']:.4f}, F1={r['f1']:.4f} | overfit={of}, gap={gap_pct:.1f}%")

import re
def _label_numeric_key(s: str) -> int:
    m = re.findall(r'\d+', s)
    return int(m[-1]) if m else 10**9

def aggregate_and_rank(comp2rows: Dict[str, List[Dict]],
                       std_auc_max: float = 0.03,
                       std_f1_max: float  = 0.04,
                       decimals: int = 3,
                       use_ascii_pm: bool = False) -> List[Dict]:
    """
    제외 규칙 적용 → 정렬 결과(남은 후보) 반환.
    제외 규칙:
      1) overfit=YES 비율 > 2/5 제외
      2) |gap| ≥ 0.15 비율 
      3) mean(AUC) < 0.75 또는 mean(F1) < 0.72 제외
    정렬:
      mean(AUC) 내림차순 → mean(F1) 내림차순 → mean(|gap|) 오름차순
      (동률 시 std(AUC)+std(F1) 합이 작은 쪽 우선)
    """
    def _fmt_mean_std(m: float, s: float) -> str:
        pm = "+/-" if use_ascii_pm else "±"
        return f"{round(m, decimals)}{pm}{round(s, decimals)}"
    agg = []
    for label, rows in comp2rows.items():
        aucs = np.array([x["auc"] for x in rows], dtype=float)
        f1s  = np.array([x["f1"]  for x in rows], dtype=float)
        gaps = np.array([x["gap"] for x in rows], dtype=float)          # 부호 유지
        overfits = np.array([x["overfit"] for x in rows], dtype=bool)

        n = len(aucs)
        mean_auc = float(np.mean(aucs))
        mean_f1  = float(np.mean(f1s))
        mean_abs_gap = float(np.mean(np.abs(gaps)))
        
        std_auc  = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0
        std_f1   = float(np.std(f1s,  ddof=1)) if len(f1s)  > 1 else 0.0
        std_gap  = float(np.std(np.abs(gaps), ddof=1)) if n > 1 else 0.0

        overfit_rate = float(np.mean(overfits))
        gap_ge_cut_rate = float(np.mean(np.abs(gaps) >= GAP_CUTOFF))

        agg.append({
            "label": label,
            "n": n,
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "auc_mean_std": _fmt_mean_std(mean_auc, std_auc),
            "mean_f1": mean_f1,
            "std_f1": std_f1,
            "f1_mean_std": _fmt_mean_std(mean_f1, std_f1),
            "mean_abs_gap": mean_abs_gap,
            "std_gap": std_gap,
            "gap_mean_std": _fmt_mean_std(mean_abs_gap, std_gap),
            "overfit_rate": overfit_rate,
            "gap_ge_15_rate": gap_ge_cut_rate
        })

    # filter exclusion rules
    filtered = [
        a for a in agg
        if a["overfit_rate"] <= (2/5) and
           a["gap_ge_15_rate"] <= (2/5) and
           a["mean_auc"] >= AUC_MIN and a["mean_f1"] >= F1_MIN and
           a["std_auc"] <= std_auc_max and
           a["std_f1"]  <= std_f1_max
    ]

    ranked = sorted(
        filtered,
        key=lambda x: (-x["mean_auc"], -x["mean_f1"], x["mean_abs_gap"], (x["std_auc"] + x["std_f1"]))
    )
    return ranked
# =================================================================

def print_ranked_remaining(ranked: List[Dict]) -> None:
    """제외 규칙 통과만 정렬 결과 요약 출력"""
    print("\n[REMAINING CANDIDATES (after seed-based filters)]")
    if not ranked:
        print(" - No candidates passed the filters.")
        return
    for i, r in enumerate(ranked, 1):
        print(f"{i}. {r['label']} | mean AUC={r['mean_auc']:.4f} (±{r['std_auc']:.4f}), "
              f"mean F1={r['mean_f1']:.4f} (±{r['std_f1']:.4f}), "
              f"mean |gap|={r['mean_abs_gap']:.3f}, "
              f"overfit_rate={r['overfit_rate']:.2f}, |gap|>=15% rate={r['gap_ge_15_rate']:.2f}")

def plot_results(comp2rows: Dict[str, List[Dict]], plot_title: str) -> None:

    labels = sorted(comp2rows.keys(),  key=_label_numeric_key)
    
    mean_aucs = [np.mean([r['auc'] for r in comp2rows[label]]) for label in labels]
    mean_f1s = [np.mean([r['f1'] for r in comp2rows[label]]) for label in labels]
    mean_gaps = [np.mean([r['gap'] * 100 for r in comp2rows[label]]) for label in labels]
    
    x = np.arange(len(labels))  
    width = 0.35  

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar plots for AUC and F1 on the left axis
    rects1 = ax1.bar(x - width/2, mean_aucs, width, label='AUC')
    rects2 = ax1.bar(x + width/2, mean_f1s, width, label='F1', color='C1')

    ax1.set_ylabel('Score')
    ax1.set_title(plot_title)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0.6, 0.9)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Create a second y-axis for the gap
    ax2 = ax1.twinx()
    ax2.set_ylabel('Gap (%)')
    ax2.set_ylim(-15, 15)
    ax2.set_yticks(range(-14, 14, 4))

    # Plot the gap on the right axis
    line = ax2.plot(x, mean_gaps, 'r-o', label='Gap (%)', marker='o')

    # # Horizontal lines for baseline scores
    ax2.axhline(y=10, color='black', linestyle='--', linewidth=1.0)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.0)
    ax2.axhline(y=-10, color='black', linestyle='--', linewidth=1.0)
    
    # legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    fig.tight_layout()
    plt.show()

def _label_numeric_key(label: str) -> float:
    """정렬용: 'Case_10'도 올바르게 정렬되게 숫자만 추출."""
    import re
    m = re.search(r'(\d+(?:\.\d+)?)', label)
    return float(m.group(1)) if m else float('inf')

def compute_case_stats(comp2rows):
    """
    comp2rows: Dict[str, List[Dict]]
      각 row: {"auc": float, "f1": float, "gap": float, "overfit": bool}
      gap는 부호 있는 값(예: 0.083 == 8.3%)라고 가정.
    return: stats(list[dict])  # label, mean/std, 비율 등
    """
    stats = []
    for label, rows in comp2rows.items():
        aucs = np.array([r["auc"] for r in rows], dtype=float)
        f1s  = np.array([r["f1"]  for r in rows], dtype=float)
        gaps = np.array([r["gap"] for r in rows], dtype=float)          # 부호 유지
        overfits = np.array([r["overfit"] for r in rows], dtype=bool)

        n = len(aucs)
        mean_auc = float(np.mean(aucs))
        mean_f1  = float(np.mean(f1s))
        std_auc  = float(np.std(aucs, ddof=1)) if n > 1 else 0.0
        std_f1   = float(np.std(f1s,  ddof=1)) if n > 1 else 0.0

        abs_gaps = np.abs(gaps)
        mean_abs_gap = float(np.mean(abs_gaps))
        std_abs_gap  = float(np.std(abs_gaps, ddof=1)) if n > 1 else 0.0

        overfit_rate = float(np.mean(overfits))
        gap_ge_15_rate = float(np.mean(abs_gaps <= 0.1)) # 최종 평가 10% 내외 

        stats.append({
            "label": label,
            "n": n,
            "mean_auc": mean_auc, "std_auc": std_auc,
            "mean_f1":  mean_f1,  "std_f1":  std_f1,
            "mean_abs_gap": mean_abs_gap, "std_abs_gap": std_abs_gap,
            "overfit_rate": overfit_rate,
            "gap_ge_15_rate": gap_ge_15_rate,
        })
    # 보기 좋게 라벨 숫자 기준 정렬
    stats.sort(key=lambda d: _label_numeric_key(d["label"]))
    return stats

import numpy as np
import matplotlib.pyplot as plt

def plot_fig1_combined(stats, title="Feature Comparison Results"):
    """
    stats: List[Dict]
      각 dict 예:
        {
          "label": str,
          "mean_auc": float, "std_auc": float,
          "mean_f1":  float, "std_f1":  float,
          "overfit_rate": float,          # 0~1
          "gap_ge_15_rate": float         # 0~1
        }
    """
    # X축 라벨/좌표
    labels = [s["label"] for s in stats]
    x = np.arange(len(labels))
    width = 0.35  # AUC/F1 막대 폭

    mean_aucs = [s["mean_auc"] for s in stats]
    std_aucs  = [s["std_auc"]  for s in stats]
    mean_f1s  = [s["mean_f1"]  for s in stats]
    std_f1s   = [s["std_f1"]   for s in stats]

    overfit_rates  = [s["overfit_rate"]   for s in stats]
    gapge15_rates  = [s["gap_ge_15_rate"] for s in stats]

    fig, ax1 = plt.subplots(figsize=(11, 6))

    # --- Bar + Errorbar (좌측 축: 점수) ---
     # --- Bar + Errorbar (좌측 축: 점수) ---
    ax1.bar(
        x - width/2, mean_aucs, width,
        yerr=std_aucs, capsize=4,
        color="#175cb1", edgecolor="none",
        label="AUC (mean±std)",
    )
    ax1.bar(
        x + width/2, mean_f1s, width,
        yerr=std_f1s, capsize=4,
        color="#f08c21", edgecolor="none",
        label="F1 (mean±std)",
    )
    
    ax1.set_title(title)
    ax1.set_ylabel("Score")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=0)  # 라벨 길면 30~45로 조정 가능
    ax1.set_ylim(0.6, 0.9)
    ax1.grid(False)
    # , which="both", linestyle="--", linewidth=0.5, axis="y")

    # --- Line (우측 축: 비율) ---
    ax2 = ax1.twinx()
    l1, = ax2.plot(x, overfit_rates, marker="o", linewidth=2, color="#7a7a7a",label="overfit_rate")
    l2, = ax2.plot(x, gapge15_rates, marker="o", linewidth=2, color="#dd37b1", label="|gap|≤10% rate")
    ax2.set_ylabel("Rate")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(False)

    # --- Legend (양 축 핸들 합치기) ---
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + [l1,l2], labels1 + labels2, loc="upper left")

    fig.tight_layout()
    plt.show()



if __name__ == "__main__":
    import torch
    DATA_PATH = os.path.join(SRC_DIR, "input_CRF.xlsx")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(DATA_PATH)

    df = pre.preprocess_raw_data(pre.load_data(DATA_PATH))  
    df = df.reset_index(drop=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    FIXED_TEST_IDX = load_or_create_test_idx(df, TARGET)

    # ===== 멀티시드 실행 =====
    comp2rows = run_multi_seed_eval(
        df, STEPS, TARGET, device,
        seeds=SEEDS, lr=1e-3, batch_size=32, epochs=100, patience=8,
        fixed_test_idx=FIXED_TEST_IDX
    )
    print_comp_scores_per_seed(comp2rows)

    # features that followed the rules
    ranked = aggregate_and_rank(comp2rows)
    print_ranked_remaining(ranked)

    # 1 순위 피처 조합 저장
    if ranked:
         best_label = ranked[0]["label"]  
         if best_label not in STEPS:  # <<< ADDED: 안전장치
             raise KeyError(f"[save] '{best_label}'가 STEPS 키에 없습니다. 현재 키: {list(STEPS.keys())[:10]}")
           
         best_features = STEPS[best_label]       # 피처 리스트
         ver_path, alias_path = sf.save_feature_list_versioned(
             best_features,
             meta={
                 "source": "feature_step.py",
                 "best_label": best_label,
                 "mean_auc": ranked[0]["mean_auc"],
                 "mean_f1": ranked[0]["mean_f1"],
                 "mean_abs_gap": ranked[0]["mean_abs_gap"],
             },
             prefix="selected_features",          
             also_update_alias=True               
         )
         print(f"[FEATURE] 저장 완료: versioned={ver_path}  alias={alias_path}")
    else:
         print("[FEATURE] 저장할 후보가 없습니다.")
    
    # Plot 
    #plot_results(comp2rows, plot_title="Feature Select")
    
    # 1) 집계
    stats = compute_case_stats(comp2rows)

    plot_fig1_combined(stats, title="Feature Comparison Results")