from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_wine
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
np.set_printoptions(precision=5,suppress=True,floatmode='maxprec_equal')
"""
課題
カーネル関数によって更新パラメータを制御（済)
degreeの値が変わらなかったらもう一度個体生成(済)
標準化したほうがいいデータとしないほうがいいデータがある
→標準化の有無を指定出来たほうがいい
irisはしないほうがいい
分類精度をちゃんと算出する
kdd99はしたほうが良かった
多項式カーネルかつ特定のパラメータセットでは計算量が膨大になる(評価困難)
初期化の工夫
"""
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
    df_train = df.sample(frac=0.05, random_state=42)
    df_test = df.sample(frac=0.05, random_state=41)
    x_trai = df_train.drop('label', axis=1)
    t_trai = df_train['label']
    x_tes = df_test.drop('label', axis=1)
    t_tes = df_test['label']
    return x_trai, t_trai, x_tes, t_tes
# データの前処理
    #classes_to_reduce = ['normal.', 'dos.', 'probe.']
  #  reduced_data = pd.concat([
       # data[data['label'] == cls].sample(frac=0.1, random_state=1)
       # if cls in classes_to_reduce else data[data['label'] == cls]
      #  for cls in data['label'].unique()
   # ])
# 特徴量とラベルに分ける

dataset_name = 'digits'  # ここを 'iris', 'wine', 'digits', 'breast_cancer' , 'kdd99'のいずれかに変える

# データセットのロード
if dataset_name == 'iris':
    dataset = load_iris()
elif dataset_name == 'wine':
    dataset = load_wine()
elif dataset_name == 'digits':
    dataset = load_digits()
elif dataset_name == 'breast_cancer':
    dataset = load_breast_cancer()
elif dataset_name == 'kdd99':
    x_train, t_train, x_test, t_test = load_kdd99()
else:
    raise ValueError("Invalid dataset name. Choose from 'iris', 'wine', 'digits', 'breast_cancer'.")

# データをトレーニングセットとテストセットに分割する

if dataset_name != 'kdd99':
    df_dataset = pd.DataFrame(data = dataset.data,columns=dataset.feature_names)
    x_train, x_test, t_train, t_test = train_test_split(
    dataset.data, dataset.target, test_size=0.3, random_state=0)

std_scaler = StandardScaler()
std_scaler.fit(x_train)  # 訓練データでスケーリングパラメータを学習
x_train_std = std_scaler.transform(x_train)  # 訓練データの標準化
x_test_std = std_scaler.transform(x_test)    # テストデータの標準化
STD = 0#0で標準化有
svm_time = 0
s_svm_time = time.perf_counter()
svc = svm.SVC()
if STD == 0:
    svc.fit(x_train_std, t_train)
    predictions = svc.predict(x_test_std)
else:
    svc.fit(x_train, t_train)
    predictions = svc.predict(x_test)
accuracy = accuracy_score(t_test, predictions)

e_svm_time = time.perf_counter()
svm_time += e_svm_time - s_svm_time

# 結果の出力
print("Fitness:", accuracy)
# 実行時間の出力
print(f"実行時間: {svm_time:.4f}秒")
