import os
import sys
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
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
# # 원시 그대로
# Case_1 = ['AGE', 'SEX', 'SBP', 'DBP', 'SMOKE', 'ALCHOL', 'PHY_ACT',
#           'GLUCOSE', 'Hb', 'CR', 'AST', 'ALT', 'RGTP', "T_CHOL", "HDL", "LDL", "TG"]
# # Hb', 'Cr' 제외
# Case_2 = ['AGE', 'SEX', 'SBP', 'DBP', 'SMOKE', 'ALCHOL', 'PHY_ACT', 
#         'GLUCOSE', 'AST', 'ALT', 'RGTP', "T_CHOL", "HDL", "LDL", "TG"]          
# # 원시 지질 + 플래그
# Case_3 = ['AGE', 'SEX', 'SBP', 'DBP', 'SMOKE', 'ALCHOL', 'PHY_ACT', 
#         'GLUCOSE','AST', 'ALT', 'RGTP', "T_CHOL", "HDL", "LDL", "TG",  
#         'extreme_GLUCOSE_flag', 'extreme_SBP_flag', 'extreme_DBP_flag']   
# # 파생 지질 + 플래그
# Case_4 = ['AGE', 'SEX', 'SBP', 'DBP', 'SMOKE', 'ALCHOL', 'PHY_ACT', 
#         'GLUCOSE','AST', 'ALT', 'RGTP', 'LDL/HDL', 'AIP', 'Non_HDL',
#         'extreme_GLUCOSE_flag', 'extreme_SBP_flag', 'extreme_DBP_flag']  
# # 파생 지질 + 플래그 + liverrisk       
# Case_5 = ['AGE', 'SEX', 'SBP', 'DBP', 'SMOKE', 'ALCHOL', 'PHY_ACT', 
#         'GLUCOSE','AST', 'ALT', 'RGTP', 'LDL/HDL', 'AIP', 'Non_HDL', 
#         'extreme_GLUCOSE_flag', 'extreme_SBP_flag', 'extreme_DBP_flag','extreme_LiverRisk_flag']            
# 원시 그대로
Case_1 = ['AGE', 'SEX', 'SBP', 'DBP', 'SMOKE', 'ALCHOL', 'PHY_ACT',
          'GLUCOSE', 'Hb', 'CR', 'AST', 'ALT', 'RGTP', "T_CHOL", "HDL", "LDL", "TG"]

# hb, cr 제외 

Case_2 = ['AGE', 'SEX', 'SBP', 'DBP', 'SMOKE', 'ALCHOL', 'PHY_ACT', 
        'GLUCOSE', 'AST', 'ALT', 'RGTP', "T_CHOL", "HDL", "LDL", "TG"]

#플래그 3

Case_3 = ['AGE', 'SEX', 'SBP', 'DBP', 'SMOKE', 'ALCHOL', 'PHY_ACT', 
        'GLUCOSE','AST', 'ALT', 'RGTP', "T_CHOL", "HDL", "LDL", "TG",  
        'extreme_GLUCOSE_flag', 'extreme_SBP_flag', 'extreme_DBP_flag']
#플래그 2

Case_4 = ['AGE', 'SEX', 'SBP', 'DBP', 'SMOKE', 'ALCHOL', 'PHY_ACT', 
        'GLUCOSE','AST', 'ALT', 'RGTP', "T_CHOL", "HDL", "LDL", "TG",  
         'extreme_SBP_flag', 'extreme_DBP_flag'] 

# 지질 파생 1

Case_5 = ['AGE', 'SEX', 'SBP', 'DBP', 'SMOKE', 'ALCHOL', 'PHY_ACT', 
        'GLUCOSE','AST', 'ALT', 'RGTP', 'LDL/HDL', 'AIP', 'Non_HDL',
        'extreme_GLUCOSE_flag', 'extreme_SBP_flag', 'extreme_DBP_flag']
# 지질 파생 2

Case_6 = ['AGE', 'SEX', 'SBP', 'DBP', 'SMOKE', 'ALCHOL', 'PHY_ACT', 
        'GLUCOSE','AST', 'ALT', 'RGTP','LDL/HDL', 'AIP', 'Non_HDL'  
        'extreme_SBP_flag', 'extreme_DBP_flag']

STEPS = {
    "Case_1": Case_1,
    "Case_2": Case_2,
    "Case_3": Case_3,
    "Case_4": Case_4,
    "Case_5": Case_5,
    "Case_6": Case_6
} 

TARGET = "PWV"  # 타깃(라벨)

# =====================[ADDED] 시드/기준 상수 =====================
SEEDS = [17, 42, 48]          # 3개 시드 129, 252
GAP_CUTOFF = 0.15               # |gap| 15% 컷(절대값 기준)
AUC_MIN, F1_MIN = 0.75, 0.72    # 베이스라인 하한
# ================================================================

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
                     check_overfit: bool = True, plot_loss: bool = False
                     ) -> Tuple[float, float, Dict]:
    """
    df: (피처 + target)만 포함한 데이터프레임
    반환: (AUC, F1, extras) — extras에는 과적합 신호/갭 등 기록
    """
    # mlp.set_seed(42)
    # data split/scaling
    train_loader, val_loader, test_loader, input_dim = mlp.data_split(
        df, target_column=target, batch_size=batch_size
    )

    import torch
    model = mlp.mlp_model(input_dim, output_dim=2).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(  # [ADDED]
    #     optimizer, mode='min', factor=0.5, patience=3
    # )

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

    # (plot 불필요)

    return float(auc), float(f1), extras

def run_steps_raw_only(df: pd.DataFrame, steps: Dict[str, List[str]], target: str, device,
                       lr=1e-3, batch_size=32, epochs=30, patience=2,
                       cumulative: bool = False, print_each: bool = True,
                       check_overfit: bool = True, plot_loss_each: bool = False):
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
            check_overfit=check_overfit, plot_loss=plot_loss_each
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
                        seeds: List[int] = SEEDS, lr=1e-3, batch_size=32, epochs=100, patience=8
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
            check_overfit=True, plot_loss_each=False
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
                       std_auc_max: float = 0.02,
                       std_f1_max: float  = 0.03) -> List[Dict]:
    """
    제외 규칙 적용 → 정렬 결과(남은 후보) 반환.
    제외 규칙:
      1) overfit=YES 비율 > 1/3 제외
      2) |gap| ≥ 0.15 비율 > 1/3 제외
      3) mean(AUC) < 0.75 또는 mean(F1) < 0.72 제외
    정렬:
      mean(AUC) 내림차순 → mean(F1) 내림차순 → mean(|gap|) 오름차순
      (동률 시 std(AUC)+std(F1) 합이 작은 쪽 우선)
    """
    agg = []
    for label, rows in comp2rows.items():
        aucs = np.array([x["auc"] for x in rows], dtype=float)
        f1s  = np.array([x["f1"]  for x in rows], dtype=float)
        gaps = np.array([x["gap"] for x in rows], dtype=float)          # 부호 유지
        overfits = np.array([x["overfit"] for x in rows], dtype=bool)

        mean_auc = float(np.mean(aucs))
        mean_f1  = float(np.mean(f1s))
        mean_abs_gap = float(np.mean(np.abs(gaps)))
        std_auc  = float(np.std(aucs, ddof=0)) if len(aucs) > 1 else 0.0
        std_f1   = float(np.std(f1s,  ddof=0)) if len(f1s)  > 1 else 0.0

        overfit_rate = float(np.mean(overfits))
        gap_ge_cut_rate = float(np.mean(np.abs(gaps) >= GAP_CUTOFF))

        agg.append({
            "label": label,
            "mean_auc": mean_auc,
            "mean_f1": mean_f1,
            "mean_abs_gap": mean_abs_gap,
            "std_auc": std_auc,
            "std_f1": std_f1,
            "overfit_rate": overfit_rate,
            "gap_ge_15_rate": gap_ge_cut_rate
        })

    # filter exclusion rules
    filtered = [
        a for a in agg
        if a["overfit_rate"] <= (1/3) and
           a["gap_ge_15_rate"] <= (1/3) and
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
    rects1 = ax1.bar(x - width/2, mean_aucs, width, label='AUC', alpha=0.7)
    rects2 = ax1.bar(x + width/2, mean_f1s, width, label='F1', alpha=0.7, color='C1')

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



if __name__ == "__main__":
    import torch
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(SRC_DIR, "input_CRF.xlsx")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(DATA_PATH)

    df = pre.preprocess_raw_data(pre.load_data(DATA_PATH))  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== 멀티시드 실행 =====
    comp2rows = run_multi_seed_eval(
        df, STEPS, TARGET, device,
        seeds=SEEDS, lr=1e-3, batch_size=32, epochs=100, patience=8
    )
    print_comp_scores_per_seed(comp2rows)

    # features that followed the rules
    ranked = aggregate_and_rank(comp2rows)
    print_ranked_remaining(ranked)

    # Plot 
    plot_results(comp2rows, plot_title="Feature Select")