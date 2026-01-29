import torch
import importlib
import torch.nn as nn
import random
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from dataclasses import dataclass
import os

from config import GlobalParams
from models.GNN_model import AttentionGCN
from utils.graph_utils import hop_index, channel_edge_index,set_seed
import dataset


sub_params=GlobalParams()
ds=dataset.get_datset(sub_params)

@dataclass
class Superparams:
    out_channels=GlobalParams.out_channels
    lr=GlobalParams.lr
    num_epoch=GlobalParams.num_epoch
    batch_size=GlobalParams.batch_size
    train_ratio=GlobalParams.train_ratio
    lambda_balance=GlobalParams.lambda_balance
    load_file=GlobalParams.load_file
    seed=GlobalParams.seed
    mask_ratio=GlobalParams.mask_ratio
    if (GlobalParams.load_file=="data.dataset_path") or(GlobalParams.load_file=="data.dataset_branch"):
        in_channels=int(ds['data_step'])
    elif (GlobalParams.load_file=="data.dataset_path_mask") or(GlobalParams.load_file=="data.dataset_branch_mask"):
        in_channels=2*int(ds['data_step'])
    volt_step=int(ds['volt_step'])
    num_nodes=int(ds['num_nodes'])
    num_data=int(ds['num_data'])
    B=ds['B']

#--parameter--
params=Superparams()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
set_seed(params.seed)

#originalの部分
#特徴量毎のグラフ定義
#kステップ目のデータ-k-hop目までつながった行列で定義
structure_dict={}
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

#--推移をみるもの--
#損失関数
train_loss_history=[]
test_loss_history=[]
#正答率
train_acc_history=[]
test_acc_history=[]
#MAE
train_MSE_history=[]
test_MSE_history=[]

#--loop--
for epoch in range(params.num_epoch):
    model.train()
    total_loss = 0
    correct = 0
    total=0
    MSE_score=0

    #--train--
    for batch in train_loader:
        batch=batch.to(device)
        optimizer.zero_grad()
        pred ,node_pred= model(batch.x,batch.edge_index,batch.batch,params) 
        y_reg=batch.y
        if y_reg.dim() == 3: y_reg = y_reg.squeeze(1)

        #回帰出力のloss
        loss_reg = criterion_reg(pred, y_reg)

        #電源位置ノードの確率に対するloss
        node_pred_reshaped=node_pred.view(-1,params.num_nodes)
        target_reshaped=batch.target_idx
        loss_class=criterion_class(node_pred_reshaped,target_reshaped)

        loss=loss_reg+params.lambda_balance*loss_class

        loss.backward()
        optimizer.step()

        #plot用の評価値
        total_loss += loss.item()

        score_matrix=node_pred.view(-1,params.num_nodes)
        pred_label=score_matrix.argmax(dim=1)
        correct+=(pred_label==batch.target_idx).sum().item()
        total+=batch.target_idx.size(0)

        MSE_score+=loss_reg.item()

    avg_train_loss=total_loss/len(train_loader)
    train_loss_history.append(avg_train_loss)
    acc=correct/total
    train_acc_history.append(acc)
    avg_MSE_score=MSE_score/len(train_loader)
    train_MSE_history.append(avg_MSE_score)

    #--test--
    model.eval()
    total_loss=0
    correct=0
    total=0
    MSE_score=0
    with torch.no_grad():
        for batch in test_loader:
            batch=batch.to(device)
            pred ,node_pred= model(batch.x,batch.edge_index,batch.batch,params) 
            y_reg=batch.y
            if y_reg.dim() == 3: y_reg = y_reg.squeeze(1)

            #回帰出力のloss
            loss_reg = criterion_reg(pred, y_reg)

            #電源位置ノードの確率に対するloss
            node_pred_reshaped=node_pred.view(-1,params.num_nodes)
            target_reshaped=batch.target_idx
            loss_class=criterion_class(node_pred_reshaped,target_reshaped)

            loss=loss_reg+params.lambda_balance*loss_class

            #評価値計算
            total_loss += loss.item()

            score_matrix=node_pred.view(-1,params.num_nodes)
            pred_label=score_matrix.argmax(dim=1)
            correct+=(pred_label==batch.target_idx).sum().item()
            total+=batch.target_idx.size(0)

            MSE_score+=loss_reg.item()

    avg_test_loss=total_loss/len(test_loader)
    test_loss_history.append(avg_test_loss)
    acc=correct/total
    test_acc_history.append(acc)
    avg_MSE_score=MSE_score/len(test_loader)
    test_MSE_history.append(avg_MSE_score)

    #print(f"Epoch {epoch+1}: Total Loss={avg_train_loss:.4f} (Test: {avg_test_loss:.4f})")

#推移結果をplot
save_dir=r"/home/ike/research/results/sort_later"
#損失関数
plt.figure()
plt.plot(train_loss_history, label='Train Loss')
plt.plot(test_loss_history, label='Test Loss')
plt.yscale('log')
plt.title("Loss Curve (Normal GNN)")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.grid(True, which="both", ls="--")
filename=f"loss_curve_{params.load_file}_normalGNN_seed{params.seed}_mask{params.mask_ratio}.png"
filepath = os.path.join(save_dir, filename)
plt.savefig(filepath)
plt.close()
#正答率
plt.figure()
plt.plot(train_acc_history,label='Train Accuracy')
plt.plot(test_acc_history,label='Test Accuracy')
plt.title("Accuracy Curve (Normal GNN)")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
filename=f"acc_curve_{params.load_file}_normalGNN_seed{params.seed}_mask{params.mask_ratio}.png"
filepath = os.path.join(save_dir, filename)
plt.savefig(filepath)
plt.close()
#MSE
plt.figure()
plt.plot(train_MSE_history, label='Train MSE')
plt.plot(test_MSE_history, label='Test MSE')
plt.yscale('log')
plt.title("MSE Curve (Normal GNN)")
plt.xlabel("epoch")
plt.ylabel("MSE")
plt.legend()
plt.grid(True, which="both", ls="--")
filename=f"MSE_curve_{params.load_file}_normalGNN_seed{params.seed}_mask{params.mask_ratio}.png"
filepath = os.path.join(save_dir, filename)
plt.savefig(filepath)
plt.close()



# 1. 保存ファイル名をデータセット名に基づいて自動生成
# 例: params.load_file が "data.dataset1" なら "model_dataset1.pth" になる
dataset_name = params.load_file.split('.')[-1] 
save_path = f'model_{dataset_name}_normalGCN_seed{params.seed}_mask{params.mask_ratio}.pth' 

# 2. モデルのパラメータ(state_dict)のみを保存
# CPU/GPUどちらでも読み込めるように、一旦CPUに移して保存するのが一般的です
torch.save(model.state_dict(), save_path)

print(f"モデルの重みを {save_path} に保存しました。")
