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

#--ファイル指定--
npz_file="data_data.dataset_path_seed_42_mask1.npz"
pth_file="pth_path/model_dataset_path_myGCN_seed42_mask1.pth"
seed=100

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


class AttentionGCN(MessagePassing):
    def __init__(self,params,union_edge_index,edge_mask):
        super().__init__(aggr='add')
        self.in_channels=params.in_channels
        self.num_edges=union_edge_index.size(1)
        self.num_unique_connection=self.num_edges

        #Xavierの初期化
        self.edge_weight_base=nn.Parameter(torch.empty(self.num_unique_connection,params.in_channels))
        nn.init.xavier_uniform_(self.edge_weight_base)
        self.register_buffer("union_edge_index",union_edge_index)
        self.register_buffer("edge_mask",edge_mask)

        #nodeの計算確率用Lin層
        self.att_lin=nn.Linear(params.in_channels,params.out_channels)
        nn.init.xavier_uniform_(self.att_lin.weight)
        nn.init.zeros_(self.att_lin.bias)

        #回帰出力計算用lin層
        self.val_lin=nn.Linear(params.in_channels,params.volt_step)
        nn.init.xavier_uniform_(self.val_lin.weight)
        nn.init.zeros_(self.val_lin.bias)


    #propagateを呼ぶと内部でcollectが使われる-edge_weightも処理される
    def forward(self,x,edge_index,batch_index,params):
        #message-passing部分
        num_graphs = x.size(0) // params.num_nodes
        w1=self.edge_mask*self.edge_weight_base
        batch_w1=w1.repeat(num_graphs,1)
        h=self.propagate(edge_index,x=x,edge_weight=batch_w1)
        
        #確率
        score=self.att_lin(h)
        attention_weights=softmax(score,batch_index)
        
        #回帰出力のための補正用lin層
        node_values=self.val_lin(h)

        #attention_weightsを確立として各nodeにかけ，node方向に集約して電圧源予測
        weighted_values=attention_weights*node_values
        output=global_add_pool(weighted_values,batch_index)
        
        return output,score,h

    def message(self,x_j,edge_weight):
        return edge_weight*x_j



#--parameter--
params=Superparams()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(seed)
#originalの部分
#特徴量毎のグラフ定義
#kステップ目のデータ-k-hop目までつながった行列で定義
structure_dict={}
if "my" in pth_file:
    if ("mask2" in npz_file) or ("mask5" in npz_file) or ("mask10" in npz_file):
        for node in range(params.in_channels):
            structure_dict[int(node/2)]=hop_index(node+1,params)
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

#--重みの可視化--

#予測値vs教師データ
def visualize_voltage_wave(model,test_loader,params):
    #--特定のデータの電源vs予測値を比較--
    batch = next(iter(test_loader))

    with torch.no_grad():
        # pred_val: 回帰予測値
        batch=batch.to(device)
        pred_val, _ ,_= model(batch.x, batch.edge_index, batch.batch,params)

    # 2. サンプルを選択 (例: バッチ内の 0番目のデータ)
    sample_idx = 0

    # Tensor -> NumPy
    actual_data = batch.y[sample_idx].cpu().numpy()
    predicted_data = pred_val[sample_idx].cpu().numpy()
    target_node = batch.target_idx[sample_idx].item()

    # 3. 棒グラフの準備
    x = np.arange(1,len(actual_data)+1)  # X軸のラベル位置 (0, 1, 2, ...)
    width = 0.35                     # 棒の太さ

    fig, ax = plt.subplots(figsize=(12, 6))

    # 4. 棒を描画 (中心から少しずらして配置)
    # 正解データ (左にずらす)
    rects1 = ax.bar(x - width/2, actual_data, width, label='Actual', color='royalblue')
    # 予測データ (右にずらす)
    rects2 = ax.bar(x + width/2, predicted_data, width, label='Predicted', color='darkorange')

    # 5. ラベルやタイトルの設定
    ax.set_ylabel('Voltage')
    ax.set_xlabel('Step')
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
    plt.imshow(A,cmap="coolwarm",vmin=-1,vmax=1)
    plt.xticks(ticks=np.arange(N), labels=np.arange(1, N + 1))
    plt.yticks(ticks=np.arange(N), labels=np.arange(1, N + 1))
    plt.colorbar(label=f"Edge Weight(Channel{ch+1})")
    plt.xlabel("To node")
    plt.ylabel("From node")
    plt.title(f"Edge Weight(ch{ch+1})")
    plt.show()

def visualize_att_lin(model,params):
    x=np.arange(1,params.in_channels+1)
    with torch.no_grad():
        att_weight=model.att_lin.weight.cpu().numpy().flatten()
        att_bias=model.att_lin.bias.cpu().numpy().flatten()
    plt.figure(figsize=(6,5))
    plt.bar(x,att_weight)
    plt.title("Attention Linear Weight")
    plt.xlabel("Channel")
    plt.ylabel("Weight")
    plt.show()

    print("=== Attention Linear Bias ===")
    print(att_bias)

def visualize_val_lin(model,params):
    K=params.in_channels
    with torch.no_grad():
        val_weight=model.val_lin.weight.cpu().numpy()
        val_bias=model.val_lin.bias.cpu().numpy().flatten()
    #プロット
    plt.figure(figsize=(10,8))
    sns.heatmap(val_weight,cmap="coolwarm",center=0,annot=False)
    plt.xticks(ticks=np.arange(1,K+1),labels=np.arange(1,K+1))
    plt.yticks(ticks=np.arange(K),labels=np.arange(0,K))
    plt.title("Value Linear Weight")
    plt.xlabel("In Channels")
    plt.ylabel("Steps")
    plt.show()

    plt.figure(figsize=(6,5))
    x=np.arange(1,len(val_bias)+1)
    plt.bar(x,val_bias)
    plt.title("Value Linear Bias")
    plt.xlabel("Steps")
    plt.ylabel("Bias")
    plt.show()

def cul_acc_and_MSE(model,test_loader,params):
    model.eval()
    total=0
    correct=0
    MSE_score=0
    with torch.no_grad():
        for batch in test_loader:
            batch=batch.to(device)
            pred ,node_pred,_= model(batch.x,batch.edge_index,batch.batch,params) 
            y_reg=batch.y
            if y_reg.dim() == 3: y_reg = y_reg.squeeze(1)

            #回帰出力のloss
            loss_reg = criterion_reg(pred, y_reg)

            score_matrix=node_pred.view(-1,params.num_nodes)
            pred_label=score_matrix.argmax(dim=1)
            correct+=(pred_label==batch.target_idx).sum().item()
            total+=batch.target_idx.size(0)

            MSE_score+=loss_reg.item()
    acc=correct/total
    avg_MSE_score=MSE_score/len(test_loader)
    print("=== Accuracy ===")
    print(acc)
    print("=== MSE ===")
    print(avg_MSE_score)

def visualize_intermediate_representation(model,test_loader,params):
    model.eval()
    batch=next(iter(test_loader))
    with torch.no_grad():
        batch=batch.to(device)
        _,node_pred,h=model(batch.x,batch.edge_index,batch.batch,params)
        att_weight=model.att_lin.weight.cpu().numpy().flatten()
    
    sample_idx=0
    node_mask=(batch.batch==sample_idx).cpu().numpy()
    h_data=h.cpu().numpy()[node_mask]

    att_sorted_indices=np.argsort(np.abs(att_weight))
    att_top2_indices=att_sorted_indices[-2:][::-1]

    score_matrix=node_pred.view(-1,params.num_nodes)
    pred_label=score_matrix.argmax(dim=1)
    label=pred_label[0].cpu().numpy()+1

    x=np.arange(1,params.num_nodes+1)
    h_no1=h_data[:,att_top2_indices[0]]
    h_no2=h_data[:,att_top2_indices[1]]

    plt.figure()
    plt.bar(x,h_no1)
    plt.title(f"Intermediate Representation(ch{att_top2_indices[0]+1}) - Predicted node:{label}")
    plt.xlabel("nodes")
    plt.ylabel("value")
    plt.show()
    plt.close()

    plt.bar(x,h_no2)
    plt.title(f"Intermediate Representation(ch{att_top2_indices[1]+1}) - Predicted node:{label}")
    plt.xlabel("nodes")
    plt.ylabel("value")
    plt.show()
    plt.close()

visualize_edge_weight(model,0,params)
visualize_edge_weight(model,1,params)
visualize_att_lin(model,params)
visualize_val_lin(model,params)
visualize_voltage_wave(model,test_loader,params)
cul_acc_and_MSE(model,test_loader,params)
visualize_intermediate_representation(model,test_loader,params)
