from dataclasses import dataclass

@dataclass
class GlobalParams:
    #datasetを決定
    load_file="data.dataset_branch"
    out_channels=1 #出力は1channel
    lr=1e-3 #学習率
    num_epoch=500 #epoch数
    batch_size=50 #バッチサイズ
    train_ratio=0.8
    #loss=Loss_reg+lambda_balance*loss_class
    #回帰用lossとlabellossの割合
    lambda_balance=0.2 
    seed=42 #randomのシード
    mask_ratio=1 #maskなしのときは1に
    mask_matrix=[0,9,26,29,41] #観測node