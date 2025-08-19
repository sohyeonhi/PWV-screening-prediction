from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def data_split(df, target_column, batch_size=32,
               test_size=0.15,       # 전체 대비 test 비율
               val_size=0.1764706,    # 남은(1-test) 대비 val 비율 = 0.15/0.85
               random_state: int = 42,
               fixed_test_idx: np.ndarray | None = None
               ):
    # 타겟 분리
    x = df.drop(columns=[target_column])
    y = df[target_column]

    # 데이터 분할 train 70%, temp (test + valid) 30%
    if fixed_test_idx is not None:
        # 고정 테스트 인덱스 사용

        N = len(df)
        test_idx = np.asarray(fixed_test_idx, dtype=int)
        if test_idx.ndim != 1 or test_idx.min() < 0 or test_idx.max() >= N:
            raise ValueError("fixed_test_idx out of range or invalid shape.")

        all_idx = np.arange(N)
        trainval_idx = np.setdiff1d(all_idx, test_idx)

        # 남은 85%에서 val_size(=0.17647) 비율로 Stratified 분할 → 전체 15%
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
        (tr_rel, va_rel), = sss_val.split(x.iloc[trainval_idx], y.iloc[trainval_idx])

        tr_idx = trainval_idx[tr_rel]
        va_idx = trainval_idx[va_rel]
        te_idx = test_idx

        X_train, y_train = x.iloc[tr_idx], y.iloc[tr_idx]
        X_val,   y_val   = x.iloc[va_idx], y.iloc[va_idx]
        X_test,  y_test  = x.iloc[te_idx], y.iloc[te_idx]

    else:
        # 기존 70/15/15 경로(그대로 유지하되 파라미터 사용)
        X_train, X_temp, y_train, y_temp = train_test_split(
            x, y, test_size=(test_size + val_size*(1-test_size)),   # = 0.30
            random_state=random_state, shuffle=True, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=random_state, shuffle=True, stratify=y_temp
        )

    scaler = StandardScaler()
    X_train_scd = scaler.fit_transform(X_train)
    X_val_scd = scaler.transform(X_val)
    X_test_scd = scaler.transform(X_test)

    # Tensor 변환
    X_train_t = torch.tensor(X_train_scd, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_scd, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_scd, dtype=torch.float32)

    y_train_t = torch.tensor(y_train.values, dtype=torch.long)
    y_val_t = torch.tensor(y_val.values, dtype=torch.long)
    y_test_t = torch.tensor(y_test.values, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, X_train_t.shape[1]

def data_split_by_index(
    df: pd.DataFrame,
    target_column: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    batch_size: int = 32,
):
    """
    K-Fold/커스텀 분할용.
    - 스케일러는 train fold에만 fit → val/test에 transform (누수 방지)
    - 숫자형만 사용, target 제외
    - torch Dataset/DataLoader 반환 스펙은 기존 data_split과 동일
    반환: (train_loader, val_loader, test_loader, input_dim)
    """

    # 숫자형만 사용
    feats = [c for c in df.columns 
             if c != target_column and (is_numeric_dtype(df[c])or is_bool_dtype(df[c]))
             ]
    if len(feats) == 0:
        raise ValueError("No numeric features after filtering."
                         f"(cols={list(df.columns)}, target={target_column})"
                         )
    
    X_df = df[feats].copy()

    for c in X_df.columns:
        if is_bool_dtype(X_df[c]) or str(X_df[c].dtype).lower() in ("boolean", "booleandtype"):
            X_df[c] = X_df[c].astype("float32").fillna(0.0)
            
    
    X = X_df.astype("float32").to_numpy(dtype=np.float32)
    y = df[target_column].values.astype(np.int64)

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_va, y_va = X[val_idx], y[val_idx]
    X_te, y_te = X[test_idx], y[test_idx]

    # 스케일링: train에 fit → 나머지에 적용
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)
    X_te = scaler.transform(X_te)

    # 텐서/로더
    X_tr_t, y_tr_t = torch.tensor(X_tr), torch.tensor(y_tr)
    X_va_t, y_va_t = torch.tensor(X_va), torch.tensor(y_va)
    X_te_t, y_te_t = torch.tensor(X_te), torch.tensor(y_te)

    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_va_t, y_va_t), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=batch_size, shuffle=False)

    input_dim = X_tr.shape[1]
    
    return train_loader, val_loader, test_loader, input_dim

def mlp_model(input_dim, output_dim=2):
    # [CHANGED]: BN 제거, Dropout 완화(0.3→0.10, 0.2→0.05) — 기준모델 안정성/해석성 강화
    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Dropout(0.10),     

        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.05),     

        nn.Linear(32, output_dim)
    )
    return model
# input: X_train_t.shape[1]

def train_model(model, criterion, optimizer, train_loader, val_loader, device, patience, num_epochs, scheduler=None):
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None  # 초기화

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.float().to(device), labels.long().to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation loss 계산
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.float().to(device), val_labels.long().to(device)
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_labels).item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # if scheduler is not None:   # [ADDED]
        #     scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early stopping 체크
        if avg_val_loss < best_val_loss - 1e-9:  
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}  # [ADDED] 안전 저장
        else:
            patience_counter += 1
            print(f"  ↳ No improvement. Patience {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered 🚨")
                break

    # 최적의 모델 파라미터로 복원
    """
    학습 중 검증 성능이 가장 좋았던 시점의 모델 파라미터를 best_model_state에 저장
    이후 epoch에서 성능이 나빠져도 계속 학습되므로, 종료 시점의 모델은 best가 아닐 수 있음
    복원 필요 
    """
    if best_model_state is not None:  # [ADDED]
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses

def plot_loss_curve(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ===== 과적합 여부 평가 유틸 =====
def evaluate_overfitting(train_losses, val_losses, window=5, gap_threshold=0.15):
    """
    [ADDED]
    간단한 과적합 휴리스틱:
      1) 마지막 window 평균 기준, Val - Train 상대 격차가 gap_threshold(%) 이상이면 과적합 경고
      2) Val 커브가 하락 멈추고 상승 전환(바닥 찍고 상승) 패턴이 있으면 경고
    """
    import numpy as np

    t = np.array(train_losses, dtype=float)
    v = np.array(val_losses, dtype=float)
    n = len(t)
    if n < max(3, window):
        print("[Overfit Check] 데이터가 부족하여 평가 생략")
        return {"overfit": False, "reason": "insufficient history"}

    # 1) 마지막 window 구간 평균 격차
    t_mean = t[-window:].mean()
    v_mean = v[-window:].mean()
    rel_gap = (v_mean - t_mean) / (abs(t_mean) + 1e-8)

    # 2) 최근 3포인트로 '상승 전환' 감지
    turn_up = False
    if n >= 3:
        turn_up = (v[-3] >= v[-2]) and (v[-1] > v[-2])

    overfit = (rel_gap >= gap_threshold) or turn_up
    reason = []
    if rel_gap >= gap_threshold:
        reason.append(f"val-train 상대 격차 {rel_gap*100:.1f}% ≥ {gap_threshold*100:.0f}%")
    if turn_up:
        reason.append("검증 손실이 최근 상승 전환 패턴")

    msg = "과적합 징후 감지" if overfit else "과적합 뚜렷하지 않음"
    print(f"[Overfit Check] {msg} | last{window}_mean: train={t_mean:.4f}, val={v_mean:.4f}, gap={rel_gap*100:.1f}%")
    if reason:
        print("  - 근거:", "; ".join(reason))

    return {"overfit": bool(overfit), "gap": float(rel_gap), "turn_up": bool(turn_up), "reason": "; ".join(reason)}
