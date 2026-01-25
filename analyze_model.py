import torch
import importlib
import torch.nn as nn
import random
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from config import GlobalParams
from models.GNN_model import AttentionGCN
from utils.graph_utils import hop_index, channel_edge_index
import dataset

#--parameter--
params=dataset.Hyperparams()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#originalの部分
#特徴量毎のグラフ定義
#kステップ目のデータ-k-hop目までつながった行列で定義
structure_dict={}
for node in range(params.in_channels):
    structure_dict[node]=hop_index(node+1,params)

union_index,union_mask=channel_edge_index(params,structure_dict)

#--model定義--
model=AttentionGCN(params,union_index,union_mask)
model=model.to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=params.lr,weight_decay=1e-4)
criterion_class=nn.CrossEntropyLoss()
criterion_reg=nn.MSELoss()

#data list作成
data_list=dataset.create_torch_data_list(dataset.ds,params,model)
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
model.load_state_dict(torch.load("model_path_no_mask.pth",map_location=torch.device('cpu')))
print("load completed")

model.eval()

#--重みの可視化--

#予測値vs教師データ
def visualize_voltage_wave(model,test_loader,params):
    #--特定のデータの電源vs予測値を比較--
    batch = next(iter(test_loader))

    with torch.no_grad():
        # pred_val: 回帰予測値
        batch=batch.to(device)
        pred_val, _ = model(batch.x, batch.edge_index, batch.batch,params)

    # 2. サンプルを選択 (例: バッチ内の 0番目のデータ)
    sample_idx = 0

    # Tensor -> NumPy
    actual_data = batch.y[sample_idx].cpu().numpy()
    predicted_data = pred_val[sample_idx].cpu().numpy()
    target_node = batch.target_idx[sample_idx].item()

    # 3. 棒グラフの準備
    x = np.arange(len(actual_data))  # X軸のラベル位置 (0, 1, 2, ...)
    width = 0.35                     # 棒の太さ

    fig, ax = plt.subplots(figsize=(12, 6))

    # 4. 棒を描画 (中心から少しずらして配置)
    # 正解データ (左にずらす)
    rects1 = ax.bar(x - width/2, actual_data, width, label='Actual', color='royalblue')
    # 予測データ (右にずらす)
    rects2 = ax.bar(x + width/2, predicted_data, width, label='Predicted', color='darkorange')

    # 5. ラベルやタイトルの設定
    ax.set_ylabel('Voltage')
    ax.set_xlabel('Channel / Step Index')
    ax.set_title(f'Prediction Comparison (Target Node: {target_node})')
    ax.set_xticks(x)             # 全ての目盛りを表示
    ax.legend()                  # 凡例を表示
    ax.grid(axis='y', linestyle='--', alpha=0.7) # 縦軸方向のグリッド線

    plt.show()


def visualize_edge_weight(model,ch,params):
    #edge_weightをmaskに適用
    with torch.no_grad():
        w=model.edge_mask*model.edge_weight_base
        ch_weight=w[:,ch].cpu().numpy()
    #edge_weightとedgeの対応先を求める
    edge_index=model.union_edge_index.cpu().numpy()
    source=edge_index[0]
    distination=edge_index[1]
    #隣接行列の枠を作成
    N=params.num_nodes
    A=np.zeros((N,N))
    #Aにedge_weightを対応させる
    for i in range((len(ch_weight))):
        u=source[i]
        v=distination[i]
        A[u,v]=ch_weight[i]
    
    #プロット
    plt.figure(figsize=(6,5))
    plt.imshow(A,cmap="coolwarm")
    plt.colorbar(label=f"Edge Weight(Channel{ch})")
    plt.xlabel("To node")
    plt.ylabel("From node")
    plt.title(f"Edge Weight(ch{ch})")
    plt.show()

def visualize_att_lin(model,params):
    x=np.arange(0,params.in_channels)
    with torch.no_grad():
        att_weight=model.att_lin.weight.cpu().numpy().flatten()
        att_bias=model.att_lin.bias.cpu().numpy().flatten()
    plt.figure(figsize=(6,5))
    plt.bar(x,att_weight)
    plt.title("Attention Linear Weight")
    plt.xlabel("Channel")
    plt.ylabel("Weight")
    plt.show()

    print(f"Attention Linear Bias:{att_bias}")

def visualize_val_lin(model,params):
    with torch.no_grad():
        val_weight=model.val_lin.weight.cpu().numpy()
        val_bias=model.val_lin.bias.cpu().numpy().flatten()
    #プロット
    plt.figure(figsize=(10,8))
    sns.heatmap(val_weight,cmap="coolwarm",center=0,annot=False)
    plt.title("Value Linear Weight")
    plt.xlabel("In Channels")
    plt.ylabel("Out Channels")
    plt.show()

    plt.figure(figsize=(6,5))
    x=np.arange(len(val_bias))
    plt.bar(x,val_bias)
    plt.title("Value Linear Bias")
    plt.xlabel("Out Channels")
    plt.ylabel("Bias")
    plt.show()

visualize_edge_weight(model,0,params)
visualize_att_lin(model,params)
visualize_val_lin(model,params)
visualize_voltage_wave(model,test_loader,params)

