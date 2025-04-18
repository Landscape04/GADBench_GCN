# main.py
import os
import time
import copy
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from data import load_and_split
from models.gcn import MultiHopGCN
from train import train_epoch, validate
from eval import evaluate
from train import EarlyStopper

def main(trial_num=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {
        'hidden_dim': 128,
        'lr': 0.005,
        'weight_decay': 1e-4,
        'patience': 10,
        'max_epochs': 100
    }
    
    results = []
    start_time = 0

    for trial in range(trial_num):
        trial_result = {'trial': trial+1, 'status': 'pending'}
        try:
            # 加载数据
            data = load_and_split('datasets/reddit.pt', seed=trial+1)
            data = data.to(device)
            
            # ========== 关键修复部分 ==========
            # 计算类别权重
            train_labels = data.y[data.train_mask]
            num_pos = train_labels.sum().item()
            num_neg = len(train_labels) - num_pos
            
            if num_pos == 0 or num_neg == 0:
                raise ValueError(f"类别不平衡异常: 正样本={num_pos}, 负样本={num_neg}")
                
            pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32, device=device)
            # ==================================
            
            # 初始化模型
            model = MultiHopGCN(in_channels=dataset.num_features,
                   hidden_channels=args.hidden,
                   out_channels=args.dim,
                   num_layers=args.layer,
                   hops=args.hops).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], 
                                        weight_decay=config['weight_decay'])
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
            # 训练循环
            stopper = EarlyStopper(
                initial_model=model,
                patience=20, 
                delta=0.00005,
                warmup_epochs=10,
                smooth_window=3
            )
            for epoch in range(config['max_epochs']):
                # 训练步骤
                model.train()
                optimizer.zero_grad()
                out = model(data.x, data.edge_index)
                loss = F.binary_cross_entropy_with_logits(
                    out[data.train_mask], 
                    data.y[data.train_mask].float(),
                    pos_weight=pos_weight
                )
                loss.backward()
                optimizer.step()
                
                # 每个epoch验证（官方频率）
                model.eval()
                with torch.no_grad():
                    val_out = model(data.x, data.edge_index)
                    val_pred = torch.sigmoid(val_out[data.val_mask])
                    val_auc = roc_auc_score(data.y[data.val_mask].cpu(), val_pred.cpu())
                    
                # 早停检查
                if stopper.step(val_auc):
                    print(f"Early stopping at epoch {epoch+1}, Best AUC: {stopper.best_metric:.4f}")
                    break
                
            # 加载最佳模型（官方恢复方式）
                if stopper.best_model:
                    model.load_state_dict(stopper.best_model)
                else:
                    print("使用最终epoch的模型参数")
                    final_model = copy.deepcopy(model)                
            # 最终测试
            test_logits = validate(model, data, device)
            test_metrics = evaluate(data.y.cpu(), test_logits, data.test_mask.cpu())
            
            trial_result.update({
                'status': 'completed',
                'test_auc': test_metrics['auc'],
                'test_ap': test_metrics['ap'],
                'test_f1': test_metrics['f1'],
                'epochs': epoch+1,
                'time': time.time()-start_time
            })
            
        except Exception as e:
            trial_result.update({
                'status': f'failed: {str(e)}',
                'time': time.time()-start_time
            })
            print(f"\n[Error] 试验 {trial+1} 失败: {str(e)}")
        
        results.append(trial_result)
    
    # 保存结果
    df = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    df.to_excel('results/results.xlsx', index=False)
    
    # 打印统计
    completed = df[df['status'] == 'completed']
    print("\n=== 最终统计 ===")
    print(f"完成试验数: {len(completed)}/{trial_num}")
    if not completed.empty:
        print(f"平均AUC: {completed['test_auc'].mean():.4f} ± {completed['test_auc'].std():.4f}")
        print(f"平均AP: {completed['test_ap'].mean():.4f} ± {completed['test_ap'].std():.4f}")
        print(f"平均F1: {completed['test_f1'].mean():.4f} ± {completed['test_f1'].std():.4f}")
        print(f"平均耗时: {completed['time'].mean():.1f}s/试验")

if __name__ == "__main__":
    main(trial_num=10)