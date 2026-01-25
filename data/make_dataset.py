import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
"""
〈入力1〉グラフ情報
接続行列、隣接行列、各edgeの特性インピーダンス、グラフ端のインピーダンス
・接続行列Bは各edgeに特性インピーダンスの重みがのっている
・隣接行列Aにも特性インピーダンスの重みがのっている
・グラフ端のインピーダンスはZ_edge
・Z_edge=[Z_alpha,nan,...,Z_beta,nan,...,Z_gamma]のようにグラフ端のインピーダンスがない所はnan

〈入力2〉実験情報
電源位置node、step_number、電圧パルス発生ステップk

〈出力〉
step×nodeのdataset

〈各変数値）
Vnode：各ノードの電圧をベクトルにしたもの
v：各nodeに入力する電圧波
"""

#接続行列Bからvを定義
#入力されるグラフと入力波のグラフvgraphの対応関係をdict形式で保存
def make_v(B):
    num_node=B.shape[0]
    num_edge=np.count_nonzero(B,axis=1)
    v_dict={}
    i=0
    for node in range(num_node):
        E=num_edge[node]
        v_dict[node]=list(range(i,i+E))
        i+=E
    return np.zeros((i)),v_dict

#入力グラフのBからvgraphのB、Bvを作る
#〇-〇-〇-〇
#↓
#〇-〇〇-〇〇-〇
def make_Bv(B):
    num_node=B.shape[0]
    B_v=[]
    for i in range(num_node):
        cols=np.where(B[i]!=0)[0]
        for j in cols:
            row=np.zeros(B.shape[1],dtype=B.dtype)
            row[j]=B[i][j]
            B_v.append(row)
    return np.array(B_v)

#Bvのグラフの隣接行列Avlink
def make_Avlink(Bv,v):
    Avlink=np.zeros((len(v),len(v)))
    Bv1=Bv.T
    for i in range(Bv1.shape[0]):
        cols=np.where(Bv1[i]!=0)[0]
        vals=Bv1[i,cols]
        Avlink[cols[0]][cols[1]]=vals[0]
        Avlink[cols[1]][cols[0]]=vals[0]
    return Avlink

#Vnode[k]=P @ v[k]
def make_P(B,v,Z_edge,v_dict):
    num_node=B.shape[0]
    num_edge=np.count_nonzero(B,axis=1)
    P=np.zeros((num_node,len(v)))
    for node in range(num_node):
        #各ノードに接続されているインピーダンス数を計算
        E=num_edge[node]
        #nodeに接続されているZ
        Z_node=B[node][B[node]!=0]

        #Pの各要素を指定
        if(E==1):
            if not np.isnan(Z_edge[node]):
                P[node][v_dict[node]]=2*Z_edge[node]/(Z_node[0]+Z_edge[node])
            if np.isnan(Z_edge[node]):
                print(f"node[{node}]:open")
                P[node][v_dict[node]]=2
        if(E==2):
            if not np.isnan(Z_edge[node]):
                P[node][v_dict[node][0]]= \
                    2*Z_edge[node]*Z_node[1]/(Z_edge[node]*Z_node[0]+Z_edge[node]*Z_node[1]+Z_node[0]*Z_node[1])
                P[node][v_dict[node][1]]= \
                    2*Z_edge[node]*Z_node[0]/(Z_edge[node]*Z_node[0]+Z_edge[node]*Z_node[1]+Z_node[0]*Z_node[1])
            if np.isnan(Z_edge[node]):
                P[node][v_dict[node][0]]=2*Z_node[1]/(Z_node[0]+Z_node[1])
                P[node][v_dict[node][1]]=2*Z_node[0]/(Z_node[0]+Z_node[1])
        if(E==3):
            if not np.isnan(Z_edge[node]):
                P[node][v_dict[node][0]]= \
                    2*Z_edge[node]*Z_node[1]*Z_node[2]/(Z_node[0]*Z_node[1]*Z_node[2]+Z_edge[node]*(Z_node[0]*Z_node[1]+Z_node[1]*Z_node[2]+Z_node[2]*Z_node[0]))
                P[node][v_dict[node][1]]= \
                    2*Z_edge[node]*Z_node[2]*Z_node[0]/(Z_node[0]*Z_node[1]*Z_node[2]+Z_edge[node]*(Z_node[0]*Z_node[1]+Z_node[1]*Z_node[2]+Z_node[2]*Z_node[0]))
                P[node][v_dict[node][2]]= \
                    2*Z_edge[node]*Z_node[0]*Z_node[1]/(Z_node[0]*Z_node[1]*Z_node[2]+Z_edge[node]*(Z_node[0]*Z_node[1]+Z_node[1]*Z_node[2]+Z_node[2]*Z_node[0]))
            if np.isnan(Z_edge[node]):
                P[node][v_dict[node][0]]=2*Z_node[1]*Z_node[2]/(Z_node[0]*Z_node[1]+Z_node[1]*Z_node[2]+Z_node[2]*Z_node[0])
                P[node][v_dict[node][1]]=2*Z_node[2]*Z_node[0]/(Z_node[0]*Z_node[1]+Z_node[1]*Z_node[2]+Z_node[2]*Z_node[0])
                P[node][v_dict[node][2]]=2*Z_node[0]*Z_node[1]/(Z_node[0]*Z_node[1]+Z_node[1]*Z_node[2]+Z_node[2]*Z_node[0])
    
    return P


#v[k+1]=A @ v[k]
def make_A(Avlink,v,v_dict,Z_edge):
    num_node=len(v_dict)
    A=np.zeros((len(v),len(v)))
    for node in range(num_node):
        node_v=v_dict[node]
        E=len(node_v)
        if(E==1):
            if not np.isnan(Z_edge[node]):
                to_node=np.where(Avlink[node_v[0]]!=0)[0]
                A[node_v,to_node]=(Z_edge[node]-Avlink[node_v[0]][to_node])/(Z_edge[node]+Avlink[node_v[0]][to_node])
            if np.isnan(Z_edge[node]):
                print(f"node[{node}]:open")
                A[node_v][to_node]=1
        if(E==2):
            to_node1=np.where(Avlink[node_v[0]]!=0)[0]
            to_node2=np.where(Avlink[node_v[1]]!=0)[0]
            nodeZ1=Avlink[node_v[0]][to_node1]
            nodeZ2=Avlink[node_v[1]][to_node2]
            if not np.isnan(Z_edge[node]):
                A[node_v[0],to_node1]=\
                    (Z_edge[node]*nodeZ2-nodeZ1*(Z_edge[node]+nodeZ2))/(Z_edge[node]*nodeZ2+nodeZ1*(Z_edge[node]+nodeZ2))
                A[node_v[0],to_node2]=\
                    (2*Z_edge[node]*nodeZ2)/(Z_edge[node]*nodeZ2+nodeZ1*(Z_edge[node]+nodeZ2))
                A[node_v[1],to_node1]=\
                    (2*Z_edge[node]*nodeZ1)/(Z_edge[node]*nodeZ2+nodeZ1*(Z_edge[node]+nodeZ2))
                A[node_v[1],to_node2]=\
                    (Z_edge[node]*nodeZ1-nodeZ2*(Z_edge[node]+nodeZ1))/(Z_edge[node]*nodeZ2+nodeZ1*(Z_edge[node]+nodeZ2))
            if np.isnan(Z_edge[node]):
                A[node_v[0],to_node1]=(nodeZ2-nodeZ1)/(nodeZ1+nodeZ2)
                A[node_v[0],to_node2]=2*nodeZ2/(nodeZ1+nodeZ2)
                A[node_v[1],to_node1]=2*nodeZ1/(nodeZ1+nodeZ2)
                A[node_v[1],to_node2]=(nodeZ1-nodeZ2)/(nodeZ1+nodeZ2)
        if(E==3):
            to_node1=np.where(Avlink[node_v[0]]!=0)[0]
            to_node2=np.where(Avlink[node_v[1]]!=0)[0]
            to_node3=np.where(Avlink[node_v[2]]!=0)[0]
            nodeZ1=Avlink[node_v[0]][to_node1]
            nodeZ2=Avlink[node_v[1]][to_node2]
            nodeZ3=Avlink[node_v[2]][to_node3]
            if not np.isnan(Z_edge[node]):
                A[node_v[0],to_node1]=\
                    (nodeZ2*nodeZ3*Z_edge[node]-nodeZ1*(nodeZ2*nodeZ3+nodeZ3*Z_edge[node]+Z_edge[node]*nodeZ2))/(nodeZ2*nodeZ3*Z_edge[node]+nodeZ1*(nodeZ2*nodeZ3+nodeZ3*Z_edge[node]+Z_edge[node]*nodeZ2))
                A[node_v[0],to_node2]=\
                    (2*nodeZ2*nodeZ3*Z_edge[node])/(nodeZ2*nodeZ3*Z_edge[node]+nodeZ1*(nodeZ2*nodeZ3+nodeZ3*Z_edge[node]+Z_edge[node]*nodeZ2))
                A[node_v[0],to_node3]=\
                    (2*nodeZ2*nodeZ3*Z_edge[node])/(nodeZ2*nodeZ3*Z_edge[node]+nodeZ1*(nodeZ2*nodeZ3+nodeZ3*Z_edge[node]+Z_edge[node]*nodeZ2))
                A[node_v[1],to_node1]=\
                    (2*nodeZ3*nodeZ1*Z_edge[node])/(nodeZ2*nodeZ3*Z_edge[node]+nodeZ1*(nodeZ2*nodeZ3+nodeZ3*Z_edge[node]+Z_edge[node]*nodeZ2))
                A[node_v[1],to_node2]=\
                    (nodeZ3*nodeZ1*Z_edge[node]-nodeZ2*(nodeZ3*nodeZ1+nodeZ1*Z_edge[node]+Z_edge[node]*nodeZ3))/(nodeZ2*nodeZ3*Z_edge[node]+nodeZ1*(nodeZ2*nodeZ3+nodeZ3*Z_edge[node]+Z_edge[node]*nodeZ2))
                A[node_v[1],to_node3]=\
                    (2*nodeZ3*nodeZ1*Z_edge[node])/(nodeZ2*nodeZ3*Z_edge[node]+nodeZ1*(nodeZ2*nodeZ3+nodeZ3*Z_edge[node]+Z_edge[node]*nodeZ2))
                A[node_v[2],to_node1]=\
                    (2*nodeZ1*nodeZ2*Z_edge[node])/(nodeZ2*nodeZ3*Z_edge[node]+nodeZ1*(nodeZ2*nodeZ3+nodeZ3*Z_edge[node]+Z_edge[node]*nodeZ2))
                A[node_v[2],to_node2]=\
                    (2*nodeZ1*nodeZ2*Z_edge[node])/(nodeZ2*nodeZ3*Z_edge[node]+nodeZ1*(nodeZ2*nodeZ3+nodeZ3*Z_edge[node]+Z_edge[node]*nodeZ2))
                A[node_v[2],to_node3]=\
                    (nodeZ1*nodeZ2*Z_edge[node]-nodeZ3*(nodeZ1*nodeZ2+nodeZ2*Z_edge[node]+Z_edge[node]*nodeZ1))/(nodeZ2*nodeZ3*Z_edge[node]+nodeZ1*(nodeZ2*nodeZ3+nodeZ3*Z_edge[node]+Z_edge[node]*nodeZ2))
            if np.isnan(Z_edge[node]):
                A[node_v[0],to_node1]=(nodeZ2*nodeZ3-nodeZ1*(nodeZ2+nodeZ3))/(nodeZ1*nodeZ2+nodeZ2*nodeZ3+nodeZ3*nodeZ1)
                A[node_v[0],to_node2]=(2*nodeZ2*nodeZ3)/(nodeZ1*nodeZ2+nodeZ2*nodeZ3+nodeZ3*nodeZ1)
                A[node_v[0],to_node3]=(2*nodeZ2*nodeZ3)/(nodeZ1*nodeZ2+nodeZ2*nodeZ3+nodeZ3*nodeZ1)
                A[node_v[1],to_node1]=(2*nodeZ3*nodeZ1)/(nodeZ1*nodeZ2+nodeZ2*nodeZ3+nodeZ3*nodeZ1)
                A[node_v[1],to_node2]=(nodeZ3*nodeZ1-nodeZ2*(nodeZ1+nodeZ3))/(nodeZ1*nodeZ2+nodeZ2*nodeZ3+nodeZ3*nodeZ1)
                A[node_v[1],to_node3]=(2*nodeZ3*nodeZ1)/(nodeZ1*nodeZ2+nodeZ2*nodeZ3+nodeZ3*nodeZ1)
                A[node_v[2],to_node1]=(2*nodeZ1*nodeZ2)/(nodeZ1*nodeZ2+nodeZ2*nodeZ3+nodeZ3*nodeZ1)
                A[node_v[2],to_node2]=(2*nodeZ1*nodeZ2)/(nodeZ1*nodeZ2+nodeZ2*nodeZ3+nodeZ3*nodeZ1)
                A[node_v[2],to_node3]=(nodeZ1*nodeZ2-nodeZ3*(nodeZ1+nodeZ2))/(nodeZ1*nodeZ2+nodeZ2*nodeZ3+nodeZ3*nodeZ1)
    return A

#v[1]=D @ Vnode[0]
#電源ノードには内部抵抗としてインピーダンスが設置されている前提
def make_D(Z_edge,Avlink,v_dict):
    num_node=len(Z_edge)
    Z_edge_clean=np.nan_to_num(Z_edge)
    D=np.zeros((Avlink.shape[0],num_node))
    for node in range(num_node):
        node_v=v_dict[node]
        E=len(node_v)
        if(E==1):
            Z=Avlink[node_v[0]][Avlink[node_v[0]]!=0]
            part=Z/(Z_edge_clean[node]+Z)
            to_node=np.where(Avlink[node_v[0]]!=0)[0]
            D[to_node,node]=part
        if(E==2):
            Z1=Avlink[node_v[0]][Avlink[node_v[0]]!=0]
            Z2=Avlink[node_v[1]][Avlink[node_v[1]]!=0]
            part=(Z1*Z2)/(Z1*Z2+Z_edge_clean[node]*(Z1+Z2))
            to_node1=np.where(Avlink[node_v[0]]!=0)[0]
            to_node2=np.where(Avlink[node_v[1]]!=0)[0]
            D[to_node1,node]=part
            D[to_node2,node]=part
        if(E==3):
            Z1=Avlink[node_v[0]][Avlink[node_v[0]]!=0]
            Z2=Avlink[node_v[1]][Avlink[node_v[1]]!=0]
            Z3=Avlink[node_v[2]][Avlink[node_v[2]]!=0]
            part=(Z1*Z2*Z3)/(Z1*Z2*Z3+Z_edge_clean[node]*(Z1*Z2+Z2*Z3+Z3*Z1))
            to_node1=np.where(Avlink[node_v[0]]!=0)[0]
            to_node2=np.where(Avlink[node_v[1]]!=0)[0]
            to_node3=np.where(Avlink[node_v[2]]!=0)[0]
            D[to_node1,node]=part
            D[to_node2,node]=part
            D[to_node3,node]=part
    return(D)

#電圧パルスの発射時刻のオフセットを与える
def pad_top(A, shift, data_len):
    """
    A: (T × N) の行列
    shift: 上方向のゼロ行数
    data_len: 出力時系列長（最終サイズ）
    """
    T, N = A.shape

    # 上に shift 行ゼロ
    padded = np.zeros((shift + T, N))
    padded[shift:] = A

    # 必要なら後ろをゼロで埋めて data_len に揃える
    if padded.shape[0] < data_len:
        out = np.zeros((data_len, N))
        out[:padded.shape[0]] = padded
        return out
    else:
        # もし padded が data_len より長いなら切り詰める
        return padded[:data_len]


#電源位置ノードと電源の大きさから0~step_numまでのノード電圧計算
def make_V_node_record(source_node,V_scale,step_num,k_pulse,B,Z_edge):
    #--P,A,D計算--
    v,v_dict=make_v(B)
    Bv=make_Bv(B)
    Avlink=make_Avlink(Bv,v)
    P=make_P(B,v,Z_edge,v_dict)
    A=make_A(Avlink,v,v_dict,Z_edge)
    D=make_D(Z_edge,Avlink,v_dict)

    num_node=len(Z_edge)
    Vnode=np.zeros(num_node)
    Vnode[source_node]=V_scale

    V_node_record = np.zeros((step_num+1, num_node))
    V_node_record[0]=Vnode.copy()
    v = D @ Vnode
    for j in range(step_num-k_pulse):
        v_new = np.zeros(len(v))
        Vnode = np.zeros(num_node)
        Vnode = P @ v
        V_node_record[j+1] = Vnode.copy()
        v_new = A.T @ v
        v = v_new.copy()
    return pad_top(V_node_record,k_pulse,step_num)


