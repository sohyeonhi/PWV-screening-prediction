# mlp_main.py — 최종 피처/임계값 적용 단일 학습·평가 실행기
from __future__ import annotations
import os, json, argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
import inspect

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score
from torch.optim import Adam, AdamW
from pathlib import Path

# 프로젝트 모듈
import preprocess as pre
import train_model as tm
import mlp_eva as eva
try:
    import feature_step as fs
except Exception:
    fs = None


# -----------------------
# 유틸
# -----------------------
def load_final_features_csv(csv_path: str) -> List[str]:
    """nested_cv_runner_* 가 저장한 final_features.csv에서 피처 리스트 추출."""
    df = pd.read_csv(csv_path)
    # 러너가 "final_features" 컬럼 1행에 '|'로 join하여 저장함
    if "final_features" in df.columns and df.shape[0] > 0:
        feats = str(df.loc[0, "final_features"]).split("|")
        feats = [f.strip() for f in feats if isinstance(f, str) and f.strip()]
        return feats
    # 혹시 "feature" 단일 컬럼(세로 나열) 형태면 이것도 허용
    if "feature" in df.columns:
        return [str(x).strip() for x in df["feature"].tolist() if str(x).strip()]
    raise ValueError(f"[final_features.csv] 형식 오류: {csv_path}")


def params_to_arch(optuna_params: dict):
    n = int(optuna_params["n_layers"])
    hidden = [int(optuna_params[f"h{i}"]) for i in range(n)]
    dropouts = [float(optuna_params[f"do{i}"]) for i in range(n)]
    return hidden, dropouts

# -----------------------
# 메인 실행
# -----------------------
def run(data_path: str,
        target: str,
        out_dir: str,
        final_features_csv: str,
        # decision_json: str | None,
        dropout: float = 0.5,
        weight_decay: float = 0.01,
        val_size: float = 0.2,
        seed: int = 42,
        params_path: str | None = None,
        epochs: int | None = None) -> dict:

    os.makedirs(out_dir, exist_ok=True)
    tm.set_seed(seed)
    # 1) 데이터 로드 & 전처리
    raw = pre.load_data(data_path)
    df  = pre.preprocess_raw_data(raw)
    if target not in df.columns:
        raise KeyError(f"target_column='{target}' 이 데이터에 없습니다. 예: {list(df.columns)[:25]} ...")
    print(df.columns)
    # 2) final_features.csv → 최종 피처 강제 적용
    feats = load_final_features_csv(final_features_csv)
    feats = [c for c in feats if c in df.columns and c != target]
    if not feats:
        raise ValueError(f"[final_features.csv]에서 유효한 피처를 찾지 못했습니다: {final_features_csv}")
    df = df[feats + [target]].copy()
    print(f"[INFO] Final features applied ({len(feats)}개): {feats}")

    # 1) Optuna 결과 로드
    best = {}
    if params_path:
        cand = Path(params_path)
        if not cand.is_file():
            cand = (Path(__file__).parent / params_path).resolve()
        params_json_path = str(cand)
    elif os.path.isfile(os.path.join(out_dir, "optuna_best_params8.json")):
        params_json_path = os.path.join(out_dir, "optuna_best_params8.json")
    else:
        params_json_path = str(Path("./results/tuning/optuna_tune_ext/optuna_best_params8.json").resolve())

    if not os.path.isfile(params_json_path):
        raise FileNotFoundError(f"[params_path] Optuna 파라미터 파일을 찾을 수 없습니다: {params_json_path}")

    print(f"[INFO] params_json_path = {params_json_path}")
    
    train_epochs = epochs if epochs is not None else int(best.get("epochs", 90))

    with open(params_json_path, "r", encoding="utf-8") as f:
        P = json.load(f) 
    
    # 4) 데이터로더 구성 (고정 Test 사용)
    hidden, drops = params_to_arch(P)
    bs = int(P.get("batch_size", 32))
    fixed_idx = fs.load_or_create_test_idx(df, target) if (fs and hasattr(fs, "load_or_create_test_idx")) else None
    
    train_loader, val_loader, test_loader, input_dim = tm.data_split(
        df, 
        target_column=target,
        batch_size=bs,
        test_size=0.15,       # 전체 대비 test 비율
        val_size=0.1764706,    # 남은(1-test) 대비 val 비율 = 0.15/0.85
        random_state=seed,
        fixed_test_idx=fixed_idx    
    )

    # 5) 모델/학습 설정 (드롭아웃, WD 고정)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        print(f"[DEBUG] train_model module path: {tm.__file__}")
    except Exception:
        pass

    input_dim = len(feats)
    model = tm.mlp_model(input_dim=input_dim, output_dim=1, hidden_dims=hidden, dropouts=drops, use_bn=True).to(device)
    # model = tm.mlp_model(**kwargs).to(device)
    criterion = nn.BCEWithLogitsLoss()
    opt_name = P.get("opt", "Adamw").lower()
    if opt_name == "adamw":
        optimizer = AdamW(model.parameters(), lr=P["lr"], weight_decay=P["weight_decay"])
    else:
        optimizer = Adam(model.parameters(), lr=P["lr"], weight_decay=P["weight_decay"])

    # 6) 학습 (early stopping)
    model, train_losses, val_losses = tm.train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        # num_epochs=P["epochs"],
        patience=8,
        min_delta=3e-4,
        num_epochs=train_epochs
    )

    # 7) 검증 성능
    loss_val, auc_val, y_true_val, y_pred_val, y_prob_val = eva.evaluate_model(
        model, val_loader, criterion, device, plot=False
    ) 
    f1_val_05 = float(f1_score(y_true_val, (y_prob_val >= 0.5).astype(int)))
    f1_val_final = None
    # if final_threshold is not None:
    #     f1_val_final = float(f1_score(y_true_val, (y_prob_val >= final_threshold).astype(int)))

    # 8) 테스트 성능(고정 Test 존재 시)
    test_auc = None; test_f1_05 = None; test_f1_final = None
    if test_loader is not None:
        test_loss, test_auc, y_true_te, y_pred_te, y_prob_te = eva.evaluate_model(
            model, test_loader, criterion, device, plot=True
        ) 
        test_f1_05 = float(f1_score(y_true_te, (y_prob_te >= 0.5).astype(int)))
        # if final_threshold is not None:
            # test_f1_final = float(f1_score(y_true_te, (y_prob_te >= final_threshold).astype(int)))
        print(f"[TEST] loss={test_loss:.4f},  AUC={test_auc:.4f}, F1@0.5={test_f1_05:.4f}"
              + (f", F1@final={test_f1_final:.4f}" if test_f1_final is not None else ""))

    # 9) 요약 저장
    summary = {
    "n_features": len(feats),
    "hidden_dims": hidden,
    "dropouts": [float(d) for d in drops],
    "optimizer": opt_name,
    "lr": float(P["lr"]),
    "weight_decay": float(P["weight_decay"]),
    "batch_size": int(bs),
    "epochs": train_epochs,
    "val_auc": float(auc_val),
    "val_f1@0.5": float(f1_val_05),
    }
    
    if f1_val_final is not None:
        summary["val_f1@final_threshold"] = float(f1_val_final)
        # summary["final_threshold"] = float(final_threshold)
    if test_auc is not None:
        summary["test_auc"] = float(test_auc)
        summary["test_f1@0.5"] = float(test_f1_05)
        if test_f1_final is not None:
            summary["test_f1@final_threshold"] = float(test_f1_final)

    with open(os.path.join(out_dir, "mlp_main_summary2.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser(description="Final features/decision 기반 최종 1회 학습·평가")
    ap.add_argument("--data_path", type=str, required=True, help="./input_CRF.xlsx")
    ap.add_argument("--target_column", type=str, default="PWV")
    ap.add_argument("--out_dir", type=str, default="./results/main/mlp_main_final")
    ap.add_argument("--final_features_csv", type=str, required=True,
            help="./results/exp_subsample03/final_features0.3.csv")
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--weight_decay", type=float, default=0.02)
    ap.add_argument("--val_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--params_path", type=str, default=None,
                help="./results/tuning/optuna_lst/optuna_best_params8.json")
    ap.add_argument("--epochs", type=int, default=90, help="학습 epoch 수 (지정 시 Optuna 결과보다 우선)")

    args = ap.parse_args()
    
            
    run(data_path=args.data_path,
        target=args.target_column,
        out_dir=args.out_dir,
        final_features_csv=args.final_features_csv,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        val_size=args.val_size,
        seed=args.seed,
        params_path=args.params_path,
        epochs=args.epochs
        )

if __name__ == "__main__":
    main()
