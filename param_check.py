from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
np.set_printoptions(precision=5,suppress=True,floatmode='maxprec_equal')
"""
課題
カーネル関数によって更新パラメータを制御（済)
degreeの値が変わらなかったらもう一度個体生成(済)
標準化の有無の指定(済)
多項式カーネルかつ特定のパラメータセットでは計算量が膨大になる(評価困難)→d(1-3)に変更(済)
ルーレット選択の実装(済)
実験を複数回やってそこから統計的な値を出す機能の追加(済)
学習セット、検証セット、テストセットでの分割(済)
複数のグラフを重ねて表示(済)

最良値を保存するプログラム()
ルーレット選択の式が少し違うかも(済)
パラメータcoef0の範囲（実験して良さそうな値出す)
分類精度をちゃんと算出する
初期化の工夫
"""
C_range = (1.0e-6, 3.5e4)#(1.0e-6, 3.5e4)
gamma_range =(1.0e-6, 32)#(1.0e-6, 32)
def load_kdd99():
    url = "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz"
    col_names = ["duration", "protocol_type", "service", "flag", "src_bytes",
                 "dst_bytes", "land", "wrong_fragment", "urgent",
                 "hot", "num_failed_logins", "logged_in", "num_compromised",
                 "root_shell", "su_attempted", "num_root",
                 "num_file_creations", "num_shells", "num_access_files",
                 "num_outbound_cmds", "is_host_login",
                 "is_guest_login", "count", "srv_count", "serror_rate",
                 "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
                 "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
                 "dst_host_count", "dst_host_srv_count",
                 "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
                 "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
                 "dst_host_serror_rate", "dst_host_srv_serror_rate",
                 "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]

    df = pd.read_csv(url, names=col_names)
    df= df.drop(['protocol_type', 'service', 'flag'], axis=1)
    df_train = df.sample(frac=0.1, random_state=42)
    df_check = df.sample(frac=0.1, random_state=41)
    df_test = df.sample(frac=0.1, random_state=40)
    x_trai = df_train.drop('label', axis=1)
    t_trai = df_train['label']
    x_ch = df_check.drop('label', axis=1)
    t_ch = df_check['label']
    x_tes = df_test.drop('label', axis=1)
    t_tes = df_test['label']
    return x_trai, t_trai, x_ch, t_ch, x_tes, t_tes

dataset_name = "kdd99"  # ここを 'iris', 'wine', 'digits', 'breast_cancer' , 'kdd99'のいずれかに変える
STD = 0#0で標準化有
std_scaler = MinMaxScaler()
#std_scaler = StandardScaler()
# データセットのロード
x_train, t_train, x_test, t_test, x_end, t_end = load_kdd99()
default_accuracy = 0.9983603902676005
# データをトレーニングセットとテストセットに分割する
std_scaler.fit(x_train)  # 訓練データでスケーリングパラメータを学習
x_train_std = std_scaler.transform(x_train)  # 訓練データの標準化
x_test_std = std_scaler.transform(x_test)    # テストデータの標準化
x_end_std = std_scaler.transform(x_end)

# 評価関数
def evaluate_function(solution,flag):
    global STD
    #print(f"評価中",solution) #デバッグ用    
    svc = svm.SVC(kernel='rbf',  C = solution[0]*(C_range[1]- C_range[0]) + C_range[0],
                  gamma = solution[1]*(gamma_range[1]- gamma_range[0]) + gamma_range[0],
                  max_iter= -1)
    selected_features = solution[2:] >= 0.5
    if np.sum(selected_features) == 0:
        return 0 
    if flag == 1:
        svc.fit(x_train_std[:, selected_features], t_train)#学習セット
        predictions = svc.predict(x_end_std[:, selected_features])
        Miss = 1 - accuracy_score(t_end, predictions)
    elif STD == 0:
        svc.fit(x_train_std[:, selected_features], t_train)#学習セット
        predictions = svc.predict(x_test_std[:, selected_features])#検証セット
        Miss = 1 - accuracy_score(t_test, predictions)
    else: print("標準化をしてください\n")

    return  1/(1+Miss)#ただの評価値
#ABCアルゴリズム
  # 初期化
solution = np.array([0.92818, 0.41584, 0.82410, 0.09607, 0.89313, 0.40156, 0.88714, 0.44627, 0.89007, 0.12300, 0.91560, 0.62145, 0.08931, 0.21920, 0.54027, 0.76712, 0.46283, 0.30774, 0.49329, 0.09119, 0.89088, 0.85179, 0.81813, 0.77971, 0.15209, 0.66523, 0.15144, 0.94665, 0.36397, 0.92031, 0.57207, 0.05173, 0.62607, 0.97314, 0.02873, 0.32291, 0.29008, 0.61521, 0.21289, 0.12474])
fitness = 0
evaluate_function(solution,0)
# 解の初期化
#ここにテストセットで分類精度を検証するプログラムを記述（これが最終的な分類精度)
fitness= evaluate_function(solution,1) 
# 結果の出力
print("Solution:", solution)
print("Fitness:", 2 - (1/fitness))