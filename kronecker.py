import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import torch
import torch.nn as nn

from config import GlobalParams
from utils.graph_utils import set_seed,hop_index,channel_edge_index
from models.GNN_model import AttentionGCN
#file指定
pth_file="pth_path/model_dataset_path_myGCN_seed42_mask1.pth"
npz_file="data.dataset_path_seed42_mask1"
seed=42

#loadするnpz指定
save_dir="data/cache"
save_path=os.path.join(save_dir,npz_file)
ds=np.load(save_path,allow_pickle=True)

@dataclass
class Superparams:
    out_channels=GlobalParams.out_channels
    lr=GlobalParams.lr
    num_epoch=GlobalParams.num_epoch
    batch_size=GlobalParams.batch_size
    train_ratio=GlobalParams.train_ratio
    lambda_balance=GlobalParams.lambda_balance
    mask_ratio=GlobalParams.mask_ratio
    if ("mask2" in npz_file) or ("mask5" in npz_file) or ("mask10" in npz_file):
        in_channels=2*int(ds['data_step'])
    else:
        in_channels=int(ds['data_step'])
    volt_step=int(ds['volt_step'])
    num_nodes=int(ds['num_nodes'])
    num_data=int(ds['num_data'])
    B=ds['B']

set_seed(seed)
params=Superparams()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

structure_dict={}
if "my" in pth_file:
    if ("mask2" in npz_file) or ("mask5" in npz_file) or ("mask10" in npz_file):
        for node in range(params.in_channels):
            structure_dict[node]=hop_index(int(node/2)+1,params)
    else:
        for node in range(params.in_channels):
            structure_dict[node]=hop_index(node+1,params)

elif "normal" in pth_file:
    common_hop=hop_index(1,params)
    for node in range(params.in_channels):
        structure_dict[node]=common_hop    

union_index,union_mask=channel_edge_index(params,structure_dict)

#--model定義--
model=AttentionGCN(params,union_index,union_mask)
model=model.to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=params.lr,weight_decay=1e-4)
criterion_class=nn.CrossEntropyLoss()
criterion_reg=nn.MSELoss()

#modelのロード
model.load_state_dict(torch.load(pth_file,map_location=torch.device('cpu')))
print("load completed")

model.eval()

#隣接行列
Adjacency_matrix={}
for ch in range(params.in_channels):
    with torch.no_grad():
        w=model.edge_mask*model.edge_weight_base
        ch_weight=w[:,ch].cpu().numpy()
    edge_index=model.union_edge_index.cpu().numpy()
    source=edge_index[0]
    distination=edge_index[1]
    #隣接行列の枠を作成
    A=np.zeros((params.num_nodes,params.num_nodes))
    #Aにedge_weightを対応させる
    for i in range((len(ch_weight))):
        u=source[i]
        v=distination[i]
        A[u,v]=ch_weight[i]
    Adjacency_matrix[ch]=A

#Attention Linear
with torch.no_grad():
    att_weight=model.att_lin.weight.cpu().numpy().flatten()

N=params.num_nodes
K=params.in_channels
Op=np.zeros((N,N*K))

for ch in range(K):
    # 出力node：i
    for i in range(N):
        #入力node：j
        for j in range(N):
            a_weight=Adjacency_matrix[ch][j,i]
            Op[i,j*K+ch]=a_weight*att_weight.flatten()

plt.figure(figsize=(12,6))
ax=sns.heatmap(Op,
               cmap="RdBu_r",
               center=0,
               cbar=True,
               square=True,
               linewidths=0.1,
               linecolor='lightgray',
               cbar_kws={"label":"Coefficient Values"})

# タイトルと軸ラベル
ax.set_title(f"GNN Linear Operator Visualization ($A \times W^T$)\nSize: {Op.shape}", fontsize=14)
ax.set_ylabel("Output Node Index ($i$)", fontsize=12)
ax.set_xlabel("Input Feature Index ($j \cdot K + k$)", fontsize=12)

# 視認性を上げるための区切り線とラベル
# X軸: ノードごとの区切り
for x in range(0, N * K + 1, K):
    ax.axvline(x, color='black', linewidth=1.5)

# X軸ラベルをノード単位で表示
tick_locs = np.arange(0, N * K, K) + K / 2
tick_labels = [f"Node {j}" for j in range(N)]
ax.set_xticks(tick_locs)
ax.set_xticklabels(tick_labels, rotation=0)

# Y軸: ノードごとの区切り（1行ごとですが、明示的に）
for y in range(N + 1):
    ax.axhline(y, color='black', linewidth=0.5)

plt.tight_layout()
plt.show()