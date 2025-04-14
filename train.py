# train.py
import torch
import torch.nn.functional as F

class EarlyStopper:
    """修复模型保存问题的早停类"""
    def __init__(self, 
                 patience=20,
                 delta=0.00005,
                 warmup_epochs=10,
                 smooth_window=3,
                 initial_model=None):  # 新增初始化参数
        self.patience = patience
        self.delta = delta
        self.warmup = warmup_epochs
        self.smooth_window = smooth_window
        
        self.counter = 0
        self.best_metric = -float('inf')
        self.metric_history = []
        
        # 初始化时保存初始模型
        self.best_model = {}
        if initial_model is not None:
            self._save_model(initial_model)

    def _save_model(self, model):
        """安全的模型保存方法"""
        self.best_model = {
            k: v.cpu().clone().detach()  # 确保数据不在计算图中
            for k, v in model.state_dict().items()
        }

    def step(self, current_metric, model=None):
        # 首次运行确保模型已保存
        if not self.best_model and model is not None:
            self._save_model(model)
            
        self.metric_history.append(current_metric)
        
        # 计算平滑指标
        smoothed = current_metric
        if len(self.metric_history) >= self.smooth_window:
            smoothed = sum(self.metric_history[-self.smooth_window:])/self.smooth_window
            
        # 更新最佳指标逻辑
        if (len(self.metric_history) >= self.warmup) and (smoothed > self.best_metric + self.delta):
            self.best_metric = smoothed
            self.counter = 0
            if model is not None:
                self._save_model(model)
        else:
            self.counter += 1
            
        # Warmup阶段不触发
        if len(self.metric_history) < self.warmup:
            return False
            
        return self.counter >= self.patience
    

def train_epoch(model, data, optimizer, criterion, device, epoch):
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x.to(device), data.edge_index.to(device))
    loss = criterion(out[data.train_mask], data.y[data.train_mask].float().to(device))
    
    # 梯度检查
    if torch.isnan(loss).any():
        raise RuntimeError(f"Epoch {epoch}: 损失值出现NaN")
    
    loss.backward()
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    
    optimizer.step()
    
    # 计算训练准确率
    with torch.no_grad():
        pred = (torch.sigmoid(out) > 0.5).long()
        correct = (pred[data.train_mask] == data.y[data.train_mask].to(device)).sum()
        acc = correct.float() / data.train_mask.sum()
    
    return loss.item(), acc.item()

def validate(model, data, device):
    model.eval()
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        return out.cpu()