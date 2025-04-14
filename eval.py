# eval.py
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

def evaluate(y_true, logits, mask):
    y = y_true[mask].numpy()
    prob = torch.sigmoid(logits[mask]).numpy()
    pred = (prob > 0.5).astype(int)
    
    # 处理全0或全1的情况
    if len(torch.unique(y_true[mask])) == 1:
        print("[Warning] 验证集只有单一类别")
        return {'auc': 0.5, 'ap': 0.5, 'f1': 0.0}
    
    return {
        'auc': roc_auc_score(y, prob),
        'ap': average_precision_score(y, prob),
        'f1': f1_score(y, pred)
    }