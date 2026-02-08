import torch
import importlib
import torch.nn as nn
import random
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from dataclasses import dataclass
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.utils import softmax


from config import GlobalParams
from utils.graph_utils import hop_index, channel_edge_index,set_seed
import dataset
from models.GNN_model import AttentionGCN

#--ファイル指定--
npz_file="data_data.dataset_branch_seed_42_mask1.npz"
pth_file="pth_sparce_data/model_dataset_branch_myGCN_seed42_[0, 24].pth"
seed=42

#loadするnpz指定
save_dir="data/cache"
save_path=os.path.join(save_dir,npz_file)

sub_params=GlobalParams()
if os.path.exists(save_path):
    print(f"-- Loading dataset (seed:{seed}) --")
    with np.load(save_path,allow_pickle=True) as data:
        loaded_dict={}
        for key in data.files:
            val=data[key]
            if val.ndim==0:
                loaded_dict[key]=val.item()
            else:
                loaded_dict[key]=val
ds=loaded_dict
masked_dataset=dataset.intoroduce_mask(ds["dataset"],sub_params)
ds["dataset"]=masked_dataset

@dataclass
class Superparams:
    out_channels=GlobalParams.out_channels
    lr=GlobalParams.lr
    num_epoch=GlobalParams.num_epoch
    batch_size=GlobalParams.batch_size
    train_ratio=GlobalParams.train_ratio
    lambda_balance=GlobalParams.lambda_balance
    mask_ratio=GlobalParams.mask_ratio
    in_channels=2*int(ds['data_step'])
    volt_step=int(ds['volt_step'])
    num_nodes=int(ds['num_nodes'])
    num_data=int(ds['num_data'])
    B=ds['B']

#--parameter--
params=Superparams()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(seed)
#originalの部分
#特徴量毎のグラフ定義
#kステップ目のデータ-k-hop目までつながった行列で定義
structure_dict={}
for node in range(params.in_channels):
        structure_dict[node]=hop_index(int(node/2)+1,params)  

union_index,union_mask=channel_edge_index(params,structure_dict)

#--model定義--
model=AttentionGCN(params,union_index,union_mask)
model=model.to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=params.lr,weight_decay=1e-4)
criterion_class=nn.CrossEntropyLoss()
criterion_reg=nn.MSELoss()

#data list作成
data_list=dataset.create_torch_data_list(ds,params,model)
#--train/test--
indices=list(range(params.num_data))
random.shuffle(indices)

train_size=int(params.num_data*params.train_ratio)
train_indices=indices[:train_size]
test_indices=indices[train_size:]

train_data=[data_list[i] for i in train_indices]
test_data=[data_list[i] for i in test_indices]

train_loader=DataLoader(train_data,batch_size=params.batch_size,shuffle=True)
test_loader=DataLoader(test_data,batch_size=params.batch_size,shuffle=True)


#modelのロード
model.load_state_dict(torch.load(pth_file,map_location=torch.device('cpu')))
print("load completed")

model.eval()
all_targets=[]
all_preds=[]
correct=0
total=0
with torch.no_grad():
    for batch in test_loader:
        batch=batch.to(device)
        pred ,node_pred= model(batch.x,batch.edge_index,batch.batch,params) 
        y_reg=batch.y
        if y_reg.dim() == 3: y_reg = y_reg.squeeze(1)

        node_pred_reshaped=node_pred.view(-1,params.num_nodes)
        target_reshaped=batch.target_idx

        score_matrix=node_pred.view(-1,params.num_nodes)
        pred_label=score_matrix.argmax(dim=1)
        correct+=(pred_label==batch.target_idx).sum().item()
        total+=batch.target_idx.size(0)
        all_targets.append(batch.target_idx.cpu())
        all_preds.append(pred_label.cpu())

acc=correct/total

# テンソルを結合して1つのNumPy配列に変換
targets_np = torch.cat(all_targets).numpy()+1
preds_np = torch.cat(all_preds).numpy()+1

# 3. 散布図の作成
plt.figure(figsize=(12, 12))


plt.scatter(targets_np, preds_np, alpha=0.5, s=30, c='blue', label='Prediction')

# 正解ライン (y=x)
plt.plot([1, 50], [1, 50], 'r--', label='Perfect Match (y=x)', linewidth=1.5)

ticks = np.arange(1, 51)

plt.xticks(ticks, rotation=90, fontsize=9) 
plt.yticks(ticks, fontsize=9)              

plt.xlim(0.5, 50.5)
plt.ylim(0.5, 50.5)

plt.title(f'Ground Truth vs Predicted Node (Acc:{acc})')
plt.xlabel('Ground Truth Node')
plt.ylabel('Predicted Node')
plt.legend(loc='upper left')

# グリッド線を細かく表示
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

plt.tight_layout() # レイアウトの自動調整
plt.show()