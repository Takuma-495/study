from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_wine
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import time
np.set_printoptions(precision=5,suppress=True,floatmode='maxprec_equal')
"""
デフォルト値を出すプログラム
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

    data_frame = pd.read_csv(url, names=col_names)
    data_frame= data_frame.drop(['protocol_type', 'service', 'flag'], axis=1)
    df_train = data_frame.sample(frac=0.1, random_state=42)
    df_check = data_frame.sample(frac=0.1, random_state=41)
    df_test = data_frame.sample(frac=0.1, random_state=40)
    x_trai = df_train.drop('label', axis=1)
    t_trai = df_train['label']
    x_ch = df_check.drop('label', axis=1)
    t_ch = df_check['label']
    x_tes = df_test.drop('label', axis=1)
    t_tes = df_test['label']
    return x_trai, t_trai, x_ch, t_ch, x_tes, t_tes
# 特徴量とラベルに分ける

dataset_name = 'kdd99'  # ここを 'iris', 'wine', 'digits', 'breast_cancer' , 'kdd99'のいずれかに変える

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
   x_train, t_train, x_test, t_test, x_end, t_end = load_kdd99()
else:
    raise ValueError("Invalid dataset name. Choose from 'iris', 'wine', 'digits', 'breast_cancer'.")

# データをトレーニングセットとテストセットに分割する

if dataset_name != 'kdd99':
    df_dataset = pd.DataFrame(data = dataset.data,columns=dataset.feature_names)
    x_train, x_test, t_train, t_test = train_test_split(
    dataset.data, dataset.target, test_size=0.3, random_state=0)

std_scaler = MinMaxScaler()
#std_scaler = StandardScaler()
std_scaler.fit(x_train)  # 訓練データでスケーリングパラメータを学習
x_train_std = std_scaler.transform(x_train)  # 訓練データの標準化
x_test_std = std_scaler.transform(x_test)    # テストデータの標準化
if dataset_name == 'kdd99':
    x_end_std = std_scaler.transform(x_end)
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
if dataset_name == 'kdd99':
    predictions =svc.predict(x_end_std)
accuracy = accuracy_score(t_end, predictions)

e_svm_time = time.perf_counter()
svm_time += e_svm_time - s_svm_time

# 結果の出力
print("Fitness:", accuracy)
# 実行時間の出力
print(f"実行時間: {svm_time:.4f}秒")
