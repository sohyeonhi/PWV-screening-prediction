from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
    print(f"x_list:",x.columns)
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

    g = torch.Generator()
    g.manual_seed(random_state)
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=batch_size, shuffle=True,
                              generator=g,
                              num_workers=0)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t),
                            batch_size=batch_size, shuffle=False,
                            num_workers=0)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t),
                             batch_size=batch_size, shuffle=False,
                            num_workers=0)

    return train_loader, val_loader, test_loader, X_train_t.shape[1]

def mlp_model(input_dim, output_dim=1, hidden_dims=None, dropouts=0.5, use_bn=True):
    """
    hidden_dims: List[int] (예: [16, 32])  ⟶ Optuna의 h0, h1, ... 에 대응
    dropouts: float 또는 List[float]       ⟶ Optuna의 do0, do1, ... 에 대응
    """
    if hidden_dims is None or len(hidden_dims) == 0:
        hidden_dims = [64, 32]

    # dropouts 길이/타입 정규화
    if isinstance(dropouts, (int, float)):
        dropout_list = [float(dropouts)] * len(hidden_dims)
    else:
        assert len(dropouts) == len(hidden_dims), "dropouts 길이가 hidden_dims와 달라요."
        dropout_list = [float(d) for d in dropouts]

    layers = []
    prev = input_dim
    for h, do in zip(hidden_dims, dropout_list):
        layers.append(nn.Linear(prev, h))
        if use_bn:
            layers.append(nn.BatchNorm1d(h, momentum=0.05))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(do))
        prev = h

    layers.append(nn.Linear(prev, output_dim))
    model = nn.Sequential(*layers)

    _init_he_hidden_xavier_out(model)  
    return model

def _init_he_hidden_xavier_out(model: nn.Module):
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not linear_layers:
        return
    last_linear = linear_layers[-1]
    
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if m is last_linear:
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        elif isinstance(m, nn.BatchNorm1d):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
def train_model(model, criterion, optimizer, train_loader, val_loader, device, 
                num_epochs, patience, min_delta: float = 1e-4):
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None  # 초기화

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                patience=2, threshold=1e-3, cooldown=1, min_lr=1e-5)
    
    for epoch in range(num_epochs + 1):
        model.train()
        running_loss = 0.0
        n_train = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.float().to(device) 
            labels = labels.float().view(-1,1).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            n_train += inputs.size(0)

        avg_train_loss = running_loss / max(1, n_train)
        train_losses.append(avg_train_loss)
        
        # Validation loss 계산
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs = val_inputs.float().to(device) 
                val_labels = val_labels.float().view(-1, 1).to(device)
                val_outputs = model(val_inputs)
                batch_loss = criterion(val_outputs, val_labels).item()
                val_loss += batch_loss * val_inputs.size(0)
                n_val += val_inputs.size(0)
                
        avg_val_loss = val_loss / max(1, n_val)
        val_losses.append(avg_val_loss)   
        
        if epoch == 0:
            ema = avg_val_loss
        else:
            beta = 0.6
            ema = beta * ema + (1- beta) * avg_val_loss
        
        monitor = ema
        
        scheduler.step(monitor)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, "
        f"Val Loss: {avg_val_loss:.4f}, lr={current_lr:.2e}")

        # Early stopping 체크
        if monitor + min_delta < best_val_loss :  
            best_val_loss = monitor
            patience_counter = 0
            best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}  
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
    if best_model_state is not None: 
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
    best_idx = int(np.argmin(v))           # v: val_losses 배열
    gap_at_best = (v[best_idx] - t[best_idx]) / (abs(t[best_idx]) + 1e-8)
    t_mean = t[-window:].mean()
    v_mean = v[-window:].mean()
    rel_gap_last = (v_mean - t_mean) / (abs(t_mean) + 1e-8)

    # 2) 최근 3포인트로 '상승 전환' 감지
    turn_up = False
    if n >= 3:
        turn_up = (v[-3] >= v[-2]) and (v[-1] > v[-2])
        
    rel_gap = min(gap_at_best, rel_gap_last)
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
