#--特徴量毎にグラフ形状が異なるGCN--channel Wise GCN
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.utils import softmax

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
        
        return output,score

    def message(self,x_j,edge_weight):
        return edge_weight*x_j