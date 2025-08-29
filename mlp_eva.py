# eva.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, classification_report

def evaluate_model(model, dataloader, criterion, device, 
                   plot: bool = True, threshold: float =0.5, threshold_mode: str = "youden",
                   labels=(0,1)):
    """
    - 손실, 정확도, AUC 계산
    - y_true, y_pred, y_prob 반환
    - plot=True면 ROC, 혼동행렬 시각화 + 분류 리포트 출력
    """
    model.eval()
    running_loss = 0.0
    n_samples = 0
    y_true, y_prob = [], []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).float().view(-1, 1)

            logits = model(X_batch)
            if logits.ndim == 1:  # (N,) -> (N,1)
                logits = logits.view(-1, 1)
            
            loss = criterion(logits, y_batch).item()
            running_loss += loss * X_batch.size(0)
            n_samples += X_batch.size(0)

            probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()

            y_true.extend(y_batch.detach().cpu().numpy().ravel().astype(int))
            y_prob.extend(probs) 

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    epoch_loss = running_loss / max(1, n_samples)    
    
    # # ROC & AUC
    
    can_plot_roc = np.unique(y_true).size == 2
    if can_plot_roc:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)
        epoch_auc = auc(fpr, tpr)
    else:
        fpr = tpr = thresholds = None
        epoch_auc = float("nan")

    if threshold_mode == "youden" and can_plot_roc:
        J = tpr - fpr
        idxs = np.where(J == J.max())[0]
        best_idx = int(idxs[np.argmin(fpr[idxs])])  # 동률 시 FPR 최소
        best_th = float(thresholds[best_idx])
    else:
        best_th = float(threshold)

    # 임계값 결정
    if threshold_mode.lower() == "youden" and can_plot_roc:
        J = tpr - fpr
        idxs = np.where(J == J.max())[0]
        best_idx = int(idxs[np.argmin(fpr[idxs])])  # 동률 시 FPR 최소
        best_th = float(thresholds[best_idx])
    else:
        best_th = float(threshold)

    # 하나의 임계값으로 최종 예측
    y_pred = (y_prob >= best_th).astype(int)

    if plot:
        # ROC
        if can_plot_roc:
            fig1 = plt.figure(figsize=(7, 5), dpi=150)
            ax1 = fig1.add_subplot(111)
            ax1.plot(fpr, tpr, lw=2, color="darkorange", label=f"ROC Curve (AUC = {epoch_auc:.3f})")
            ax1.plot([0, 1], [0, 1], "k--", lw=1)
            if threshold_mode.lower() == "youden":
                ax1.scatter(fpr[best_idx], tpr[best_idx], s=40)
                ax1.annotate(f"th={best_th:.3f}",
                             (fpr[best_idx], tpr[best_idx]),
                             textcoords="offset points", xytext=(8, -10),
                             fontsize=14)
            ax1.set_xlim(0, 1); ax1.set_ylim(0, 1.05)
            ax1.set_xlabel("False Positive Rate",fontsize=14); ax1.set_ylabel("True Positive Rate",fontsize=14)
            ax1.set_title("Receiver Operating Characteristic (ROC) Curve",fontsize=14)
            ax1.tick_params(axis='both', labelsize=12)
            ax1.legend(loc="lower right", fontsize=14); ax1.grid(True)
            fig1.tight_layout(); plt.show()
        else:
            print("[info] 단일 클래스라 ROC/AUC 생략합니다.")

        # Confusion Matrix

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        
        disp.plot(cmap="Blues", values_format="d")
        disp.ax_.set_xlabel("Predicted label", fontsize=14)
        disp.ax_.set_ylabel("True label", fontsize=14)
        
        for text in disp.ax_.texts:
            text.set_fontsize(14) 
        plt.title("Confusion Matrix")
        plt.grid(False)
        plt.show()

        # Text Report
        print("\n--- Model Performance ---")
        print(f"Loss: {epoch_loss:.4f}")
        print(f"AUC: {epoch_auc:.4f}")
        print(f"Threshold ({threshold_mode}): {best_th:.4f}")
        print(classification_report(y_true, y_pred, digits=4))

    return epoch_loss, epoch_auc, y_true, y_pred, y_prob

if __name__ == "__main__":
    # 외부 학습 스크립트에서 import 후 호출 권장.
    pass
