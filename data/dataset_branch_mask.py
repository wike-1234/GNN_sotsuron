import numpy as np
from data import make_dataset
import random

def generate_data(seed,mask_ratio):
    random.seed(seed)
    np.random.seed(seed)
    #マスクがランダムでなく規則的になるよう変更
    #電源ノードインピーダンスを変更してデータ数を稼ぐ
    #stepは1-50で固定

    #定数を定義
    g=3*10**8 #波の速度(光速)
    l=30 #伝送線路長
    T=l/g #周期
    num_node=50 #ノードの数
    num_edge=49 #枝の数

    data_step=50 #dataのstep数
    volt_step=50


    #接続行列の指定 
    B=np.zeros((num_node,num_edge))
    Z=np.random.uniform(49,51,num_edge) #各線路の特性インピーダンス
    for i in range(0,16):
        B[i][i]=Z[i]
        B[i+1][i]=Z[i]
    B[9][16]=Z[16]
    B[17][16]=Z[16]
    for i in range(17,34):
        B[i][i]=Z[i]
        B[i+1][i]=Z[i]
    B[29][34]=Z[34]
    B[35][34]=Z[34]
    for i in range(35,38):
        B[i][i]=Z[i]
        B[i+1][i]=Z[i]   
    B[26][38]=Z[38]
    B[39][38]=Z[38]
    for i in range(39,45):
        B[i][i]=Z[i]
        B[i+1][i]=Z[i]  
    B[41][45]=Z[45]
    B[46][45]=Z[45]
    for i in range(46,49):
        B[i][i]=Z[i]
        B[i+1][i]=Z[i] 

    #グラフ端のインピーダンスベクトル
    #グラフ端＋電源ノードにインピーダンス接続
    Z_edge_base=np.full(num_node,np.nan)
    edge_Z=50 #グラフ端のインピーダンス
    Z_edge_base[0]=edge_Z
    Z_edge_base[16]=edge_Z
    Z_edge_base[34]=edge_Z
    Z_edge_base[38]=edge_Z
    Z_edge_base[45]=edge_Z
    Z_edge_base[49]=edge_Z


    #パルスの大きさ
    V_scale_TDR=10 

    #TDR電源が端にあるときの各ノードの電源
    #最初に電源が到達してから10step分をデータとする
    V_source_base=make_dataset.make_V_node_record(0,V_scale_TDR,2*data_step,0,B,Z_edge_base)
    target_rows=data_step
    result_list=[]
    for i in range(num_node):
        col=V_source_base[:,i]
        nz_indices = np.nonzero(col)[0]
        if len(nz_indices) > 0:
            start_idx = nz_indices[0]
            # 開始位置からターゲット数分だけスライス
            extracted = col[start_idx : start_idx + target_rows]
        else:
            # 全て0の場合は、0だけの配列を用意
            extracted = np.zeros(0)
    
        # 長さが10に満たない場合、後ろに0をパディングする
        if len(extracted) < target_rows:
            padding_len = target_rows - len(extracted)
            extracted = np.pad(extracted, (0, padding_len), 'constant', constant_values=0)
        
        result_list.append(extracted)

    # 列を結合して新しい行列を作成
    V_source= np.column_stack(result_list)


    Z_variation=200 #電源位置のインピーダンスの大きさの種類
    mask_variation=1 #maskのバリエーションがmask_ratioと一致
    num_data=num_node*mask_variation*Z_variation #dataの数

    dataset = {} 
    source_node_array={}
    voltage_source={}

    mask_base=np.arange(0,num_node,mask_ratio)
    brench_node=[9,26,29,41]
    select_mask=np.union1d(mask_base,brench_node)
    all_indices=np.arange(num_node)
    mask_idx=np.setdiff1d(all_indices, select_mask)
    mask_layer=np.zeros(num_node)
    mask_layer[mask_idx]=1


    for j in range(Z_variation):
        #node_Z=49+2*j/Z_variation
        node_Z=random.randint(0,100)
        for k in range(num_node):
            Z_edge=Z_edge_base.copy()
            Z_edge[k]=node_Z
            V_node_record=np.zeros((data_step+1,num_node))

            for step in range(data_step):
                V_node_record+=make_dataset.make_V_node_record(k,V_source[step,k],data_step+1,step,B,Z_edge)
        
            # 一旦すべてのノードデータを保持する一時的な配列を作成
            masked_record = np.zeros_like(V_node_record)

            # select_maskに含まれる列（ノード）だけ、元の値をコピーする
            masked_record[:, select_mask] = V_node_record[:, select_mask]

            # 最終的なデータとして使用]
            data = masked_record[1:1+data_step]
            masked_data=np.empty((2*data_step,num_node))
            masked_data[0::2]=data
            masked_data[1::2]=mask_layer
            dataset[k+j*num_node]=masked_data
            source_node_array[k+j*num_node]=k
            voltage_source[k+j*num_node]=V_source[:,k]
    return {
    "dataset":dataset,
    "source_node_array":source_node_array,
    "voltge_source":voltage_source,
    "num_data":num_data,
    "data_step":data_step,
    "volt_step":volt_step,
    "num_nodes":num_node,
    "B":B
}