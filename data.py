# data.py
import importlib
import torch
from torch_geometric.data import Data

def allow_torch_geometric_globals():
    """安全注册PyG需要的全局类"""
    try:
        storage_module = importlib.import_module('torch_geometric.data.storage')
        classes = [
            getattr(storage_module, 'GlobalStorage'),
            getattr(storage_module, 'NodeStorage'),
            getattr(storage_module, 'EdgeStorage'),
        ]
        torch.serialization.add_safe_globals(classes)
        return True
    except Exception as e:
        print(f"[Warning] 安全类注册失败: {str(e)}")
        return False

def load_and_split(data_path, train_ratio=0.4, seed=42):
    """加载并分割数据集"""
    # 尝试安全加载模式
    security_enabled = allow_torch_geometric_globals()
    
    try:
        data_dict = torch.load(data_path, map_location='cpu', 
                              weights_only=security_enabled)
    except Exception as e:
        print(f"[Info] 安全加载失败 ({str(e)}), 尝试非安全模式")
        data_dict = torch.load(data_path, map_location='cpu', weights_only=False)

    # 数据验证
    required_keys = ['x', 'edge_index', 'y']
    assert all(k in data_dict for k in required_keys), "数据集缺少必要字段"
    
    x = data_dict['x'].float()
    edge_index = data_dict['edge_index'].long()
    y = data_dict['y'].long()
    
    num_nodes = x.size(0)
    indices = torch.randperm(num_nodes, generator=torch.Generator().manual_seed(seed))
    
    # 分割比例: 40%训练，30%验证，30%测试
    train_size = int(num_nodes * train_ratio)
    val_size = (num_nodes - train_size) // 2
    
    train_mask = torch.zeros(num_nodes, dtype=bool)
    val_mask = torch.zeros(num_nodes, dtype=bool)
    test_mask = torch.zeros(num_nodes, dtype=bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size+val_size]] = True
    test_mask[indices[train_size+val_size:]] = True

    # 打印数据统计
    print("\n=== 数据集统计 ===")
    print(f"节点数: {num_nodes}")
    print(f"边数: {edge_index.shape[1]}")
    print(f"特征维度: {x.shape[1]}")
    print(f"异常比例: {y.float().mean().item():.2%}")
    print(f"训练/验证/测试样本数: {train_mask.sum().item()}/{val_mask.sum().item()}/{test_mask.sum().item()}")

    return Data(x=x, edge_index=edge_index, y=y, 
               train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)