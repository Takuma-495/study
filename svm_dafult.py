from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn import svm
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
    df_train = df.sample(frac=0.1, random_state=42)
    df_check = df.sample(frac=0.1, random_state=41)
    df_test = df.sample(frac=0.1, random_state=39)
    x_trai = df_train.drop('label', axis=1)
    t_trai = df_train['label']
    x_ch = df_check.drop('label', axis=1)
    t_ch = df_check['label']
    x_tes = df_test.drop('label', axis=1)
    t_tes = df_test['label']
    return x_trai, t_trai, x_ch, t_ch, x_tes, t_tes

x_train, t_train, x_test, t_test, x_end, t_end = load_kdd99()

std_scaler = MinMaxScaler()
std_scaler.fit(x_train)  # 訓練データでスケーリングパラメータを学習
x_train_std = std_scaler.transform(x_train)  # 訓練データの正規化
x_test_std = std_scaler.transform(x_test)    # テストデータの正規化
x_end_std = std_scaler.transform(x_end)

STD = 0#0で正規化有
svm_time = 0
s_svm_time = time.perf_counter()
svc = svm.SVC(kernel='poly')#カーネル関数を指定
if STD == 0:
    svc.fit(x_train_std, t_train)
    predictions = svc.predict(x_test_std)
else:
    svc.fit(x_train, t_train)
    predictions = svc.predict(x_test)

predictions =svc.predict(x_end_std)
accuracy = accuracy_score(t_end, predictions)

e_svm_time = time.perf_counter()
svm_time += e_svm_time - s_svm_time
# 正解ラベルと予測ラベルを「通常状態」と「攻撃状態」に再分類
def map_labels(y):
    return ['normal' if label == 'normal' else 'attack' for label in y]

# 再分類されたラベル
y_test_mapped = map_labels(t_end)
y_pred_mapped = map_labels(predictions)

# 混同行列の計算
cm = confusion_matrix(y_test_mapped, y_pred_mapped, labels=['normal', 'attack'])

# 混同行列から各値を抽出
TN, FP, FN, TP = cm.ravel()

print(f"True Positives (TP): {TP}")
print(f"True Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")

# 検知率（再現率）
detection_rate = recall_score(y_test_mapped, y_pred_mapped, pos_label='attack')

# 誤警報率
false_alarm_rate = FP / (TN + FP)

# 適合率
precision = precision_score(y_test_mapped, y_pred_mapped, pos_label='attack')

# F値
f1 = f1_score(y_test_mapped, y_pred_mapped, pos_label='attack')

print(f"検知率: {detection_rate:.4f}")
print(f"誤警報率: {false_alarm_rate:.4f}")
print(f"適合率: {precision:.4f}")
print(f"F1-Score: {f1:.4f}")

# t_end と predictions が pandas.Series の場合に備えて iloc を使用
incorrect_indices = [i for i in range(len(t_end)) if t_end.iloc[i] != predictions[i]]

# 誤分類されたラベルを取得
incorrect_labels = [(t_end.iloc[i], predictions[i]) for i in incorrect_indices]

# 誤分類されたラベルを正解ラベルでソート
sorted_incorrect_labels = sorted(incorrect_labels, key=lambda x: x[0])

# ソート後の結果を表示
print("誤分類したデータ (正解ラベルでソート):")
for true_label, pred_label in sorted_incorrect_labels:
    print(f"True: {true_label}, Predicted: {pred_label}")

# 誤分類された "正解ラベル" の数をカウント
true_label_counts = pd.Series([true_label for true_label, pred_label in incorrect_labels]).value_counts()

# 誤分類された "予測ラベル" の数をカウント
pred_label_counts = pd.Series([pred_label for true_label, pred_label in incorrect_labels]).value_counts()

# 結果の表示 (各クラスの誤分類数)
print("\nMisclassified True Labels:")
print(true_label_counts)

print("\nMisclassified Predicted Labels:")
print(pred_label_counts)

# 結果の出力
print("Fitness:", accuracy)
# 実行時間の出力
print(f"実行時間: {svm_time:.4f}秒")
