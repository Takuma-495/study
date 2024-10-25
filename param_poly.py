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
    label_map = {
            'normal.': 'normal',
            'back.': 'DoS', 'land.': 'DoS', 'neptune.': 'DoS', 'pod.': 'DoS', 'smurf.': 'DoS', 'teardrop.': 'DoS',
            'ipsweep.': 'Probe', 'nmap.': 'Probe', 'portsweep.': 'Probe', 'satan.': 'Probe',
            'ftp_write.': 'R2L', 'guess_passwd.': 'R2L', 'imap.': 'R2L', 'multihop.': 'R2L', 'phf.': 'R2L', 'spy.': 'R2L', 'warezclient.': 'R2L', 'warezmaster.': 'R2L',
            'buffer_overflow.': 'U2R', 'loadmodule.': 'U2R', 'perl.': 'U2R', 'rootkit.': 'U2R'
        }

    # ラベルをマッピング
    df['label'] = df['label'].map(label_map)

    # 各クラスから学習、検証、テストのサンプル数を指定
    num_train = {'normal': 9746, 'DoS': 39158, 'Probe': 381, 'R2L': 112, 'U2R': 5}  # 各クラスからの学習セット数
    num_val = {'normal': 9723, 'DoS': 39112, 'Probe': 438, 'R2L': 125, 'U2R': 5}        # 各クラスからの検証セット数
    num_test = {'normal': 9781, 'DoS': 39101, 'Probe': 397, 'R2L': 117, 'U2R': 6}       # 各クラスからのテストセット数

    # 学習セット、検証セット、テストセットの抽出
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    df_test = pd.DataFrame()

    for label in num_train.keys():
        df_label = df[df['label'] == label]
        
        # 学習セット
        df_train = pd.concat([df_train, df_label.sample(n=min(num_train[label], len(df_label)), random_state=42)])
        
        # 検証セット
        df_val = pd.concat([df_val, df_label.sample(n=min(num_val[label], len(df_label)), random_state=43)])
        
        # テストセット
        df_test = pd.concat([df_test, df_label.sample(n=min(num_test[label], len(df_label)), random_state=44)])
    x_train = df_train.drop('label', axis=1)  # 特徴データ
    t_train = df_train['label']               # ラベル

    x_check = df_val.drop('label', axis=1)
    t_check = df_val['label']

    x_test = df_test.drop('label', axis=1)
    t_test = df_test['label']
    return x_train, t_train, x_check, t_check, x_test, t_test
dataset_name = "kdd99"  # ここを 'iris', 'wine', 'digits', 'breast_cancer' , 'kdd99'のいずれかに変える
STD = 0#0で標準化有
std_scaler = MinMaxScaler()
#std_scaler = StandardScaler()
# データセットのロード
x_train, t_train, x_test, t_test, x_end, t_end = load_kdd99()
DEFAULT_ACCURACY = 0.9983603902676005
# データをトレーニングセットとテストセットに分割する
std_scaler.fit(x_train)  # 訓練データでスケーリングパラメータを学習
x_train_std = std_scaler.transform(x_train)  # 訓練データの標準化
x_test_std = std_scaler.transform(x_test)    # テストデータの標準化
x_end_std = std_scaler.transform(x_end)

# 評価関数
def evaluate_function(solution,flag):
    global STD
    #print(f"評価中",solution) #デバッグ用    
    svc = svm.SVC(kernel='poly',  
                  gamma = solution[0]*(gamma_range[1]- gamma_range[0]) + gamma_range[0],
                  max_iter=int(1.0e5))
    if flag == 1:
        svc.fit(x_train_std, t_train)#学習セット
        predictions = svc.predict(x_end_std)
        Miss = 1 - accuracy_score(t_end, predictions)
    elif STD == 0:
        svc.fit(x_train_std, t_train)#学習セット
        predictions = svc.predict(x_test_std)#検証セット
        Miss = 1 - accuracy_score(t_test, predictions)
    else: print("標準化をしてください\n")

    return  1/(1+Miss)#ただの評価値
#ABCアルゴリズム
  # 初期化
solution = np.array([0])
fitness = 0
evaluate_function(solution,0)
# 解の初期化
#ここにテストセットで分類精度を検証するプログラムを記述（これが最終的な分類精度)
fitness= evaluate_function(solution,1) 
# 結果の出力
print("Solution:", solution)
print("Fitness:", 2 - (1/fitness))