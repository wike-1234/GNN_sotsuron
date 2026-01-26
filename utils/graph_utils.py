import numpy as np
import torch
import random

#グラフに対し、各ノードからnum_hop-hop離れたグラフが全て繋がっているグラフを新規で定義
def hop_index(num_hop,params):
    #重みを除く
    B_link=(params.B!=0).astype(int)
    #接続行列→隣接行列
    N,E=B_link.shape
    A=np.zeros((N,N)) 
    for i in range(E):
        connected_index=np.where(B_link[:,i]==1)[0]
        if len(connected_index)==2:
            v1,v2=connected_index
            A[v1,v2]=1
            A[v2,v1]=1
        else:
            print("Error:Edgeに接続されているnode数が2でありません")
    
    #隣接行列からk-hopまでのindexを取得
    edges=[]
    A_pow=np.eye(N)
    A_hop=np.zeros((N,N))
    for i in range(1,num_hop+1):
        A_pow=np.dot(A_pow,A)
        A_hop+=A_pow
    A_hop+=np.eye(N)
    for node_idx in range(params.num_nodes):
        row=A_hop[node_idx,:]
        hop_indices=np.where(row!=0)[0].tolist()
        for i in range(len(hop_indices)):
            edges.append([hop_indices[i],node_idx])
    return edges

#dictに入ってるindexからmask作成
def channel_edge_index(params,channel_edge_dict):
    #全てのノードが接続しているedge set作る
    unique_forward_edges_set=set()
    for ch_idx, edges in channel_edge_dict.items():
        for u,v in edges:
            if u <= v:
                unique_forward_edges_set.add((u,v))
            else:
                unique_forward_edges_set.add((v,u))
    
    forward_edges_list=sorted(list(unique_forward_edges_set))

    unique_edges_list=[]
    for u,v in forward_edges_list:
        unique_edges_list.append((u,v))
        unique_edges_list.append((v,u))
    

    num_unique_edges=len(unique_edges_list)

    #PyG用のedge_index
    unique_edge_index=torch.tensor(unique_edges_list,dtype=torch.long).t().contiguous()

    #マスク行列--マスクで特徴量毎のグラフを作成
    edge_mask=torch.zeros((num_unique_edges,params.in_channels),dtype=torch.float32)

    #エッジ検索用のマップ
    edge_map={edge:i for i,edge in enumerate(unique_edges_list)}

    #各チャネルごとに指定からmaakに1を立てる
    for ch_idx,edges in channel_edge_dict.items():
        if ch_idx>=params.in_channels:continue
        for u,v in edges:
            if(u,v) in edge_map:
                idx=edge_map[(u,v)]
                edge_mask[idx,ch_idx]=1
    
    return unique_edge_index,edge_mask

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    
