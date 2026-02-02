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
pth_file="pth_compare/model_dataset_branch_myGCN_seed42_mask1_weight_seed17.pth"
npz_file="data_data.dataset_branch_seed_42_mask1.npz"
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
    W=model.att_lin.weight.cpu().numpy().flatten()

A_stack=np.array([Adjacency_matrix[ch] for ch in range(params.in_channels)])
Weighted_A=A_stack*W[:,None,None]
Weighted_A=Weighted_A.transpose(2,1,0)
N=params.num_nodes
K=params.in_channels
Op=Weighted_A.reshape(N,N*K)

#dataが多すぎるのでinput_nodeを絞る
target_input_node=0

start_col=target_input_node*K
end_col=(target_input_node+1)*K
Op_focused=Op[:,start_col:end_col]

H, W_mat = Op_focused.shape
plt.figure(figsize=(12,12))
v_max=np.percentile(np.abs(Op_focused),99)
ax=sns.heatmap(Op_focused,
               cmap="RdBu_r",
               center=0,
               vmax=v_max,vmin=-v_max,
               cbar=True,
               square=False,
               linewidths=0.0,
               linecolor='lightgray',
               cbar_kws={"label":"Coefficient Values"})

# タイトルと軸ラベル
ax.set_title(f"GNN Linear Operator Visualization (Input Node:{target_input_node+1})", fontsize=14)
ax.set_ylabel("Output Node Index", fontsize=12)
ax.set_xlabel(f"Input Feature Index (Node:{target_input_node+1})", fontsize=12)

# X軸ラベル: 特徴量(チャンネル)のインデックス
ax.set_xticks(np.arange(W_mat) + 0.5)
ax.set_xticklabels([f"Ch {k+1}" for k in range(K)], rotation=45)

# Y軸ラベル: 出力ノード番号 (数が多い場合は間引く)
ax.set_yticks(np.arange(H) + 0.5)
ax.set_yticklabels(np.arange(H)+1, rotation=0)

plt.tight_layout()
plt.show()