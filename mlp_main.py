# mlp_main.py
import os
import torch
import torch.nn as nn
import argparse
import numpy as np

try:
    from .preprocess import load_data, preprocess_raw_data
    # from .mlp import data_split, mlp_model, train_model, plot_loss_curve, set_seed
    from .basemodel import data_split, mlp_model, train_model, plot_loss_curve, set_seed
    from .mlp_eva import evaluate_model
    from .feature_store import load_feature_list
except ImportError:
    from preprocess import load_data, preprocess_raw_data
    # from mlp import data_split, mlp_model, train_model, plot_loss_curve, set_seed
    from basemodel import data_split, mlp_model, train_model, plot_loss_curve, set_seed
    from mlp_eva import evaluate_model
    from feature_store import load_feature_list
    from sklearn.model_selection import StratifiedShuffleSplit

# ← 추가: src 디렉터리 기준 데이터 경로
SRC_DIR = os.path.dirname(os.path.abspath(__file__))   
ART_DIR = os.path.join(SRC_DIR, "artifacts")
os.makedirs(ART_DIR, exist_ok=True)
TEST_IDX_PATH = os.path.join(ART_DIR, "fixed_test_idx.npy")        
DATA_PATH = os.path.join(SRC_DIR, "input_CRF.xlsx")            

def _make_holdout_indices(df, target, test_size=0.15, random_state=42):
    y = df[target].values
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    (train_idx, test_idx), = splitter.split(df, y)
    return test_idx

def _load_or_create_test_idx(df, target):
    if os.path.exists(TEST_IDX_PATH):
        idx = np.load(TEST_IDX_PATH)
        assert idx.ndim == 1 and idx.max() < len(df) and idx.min() >= 0
        return idx
    idx = _make_holdout_indices(df, target, test_size=0.15, random_state=42)
    np.save(TEST_IDX_PATH, idx)
    return idx

# 전처리 이후 '선택 저장된 피처만' 강제 적용 함수
def enforce_selected_features(df, target: str, feature_file: str | None = None):
    """
    feature_file이 None이면 alias(selected_features.json) 우선, 없으면 최신 버전 자동탐색.
    반환: (df_filtered, feature_order, meta)
    """
    features, meta = load_feature_list(name=feature_file)
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(
            f"[FEATURE] 저장된 피처가 전처리 결과에 없습니다: {missing}\n"
            f"- 전처리 파이프라인 변경 여부 확인\n"
            f"- `python -m src.feature_step`로 최신 피처 재생성"
        )
    # 입력 순서 고정 + 타깃 결합
    filtered = df[features + [target]]
    return filtered, features, meta

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--feature-file", type=str, default=os.getenv("FEATURE_FILE"))
    args, _ = parser.parse_known_args()
    
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Data path: {DATA_PATH}")

    # 3) 데이터 로드 & 전처리
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {DATA_PATH}")
    df = load_data(DATA_PATH)
    df = preprocess_raw_data(df)
    df = df.reset_index(drop=True)
    print(f"all column:",df.columns)

    TARGET = "PWV"
    df, feature_order, meta = enforce_selected_features(df, target=TARGET, feature_file=args.feature_file)
    print(f"[FEATURE] 적용 피처 {len(feature_order)}개, fingerprint={meta.get('fingerprint')}")
    print(f"target:", df.columns)

    # 4) 데이터 분할
        
    FIXED_TEST_IDX = _load_or_create_test_idx(df,TARGET)

    train_loader, val_loader, test_loader, input_dim = data_split(
        df, target_column=TARGET, batch_size=32,
        test_size=0.15, val_size=0.1764706, random_state=42,
        fixed_test_idx=FIXED_TEST_IDX
    )

    # 5) 모델/손실/옵티마이저
    model = mlp_model(input_dim, output_dim=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 6) 학습 (Early Stopping 포함)
    model, train_losses, val_losses = train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=100,
        patience=8,
    )

    # 7) 학습 곡선
    plot_loss_curve(train_losses, val_losses)

    # 8) 테스트 평가
    test_loss, test_acc, test_auc, y_true, y_pred, y_prob = evaluate_model(
        model, test_loader, criterion, device, plot=True
    )
    print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}")

if __name__ == "__main__":
    main()
