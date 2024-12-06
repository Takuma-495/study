from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
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
kernels = [1,2,3,4]#[1, 2, 3, 4]
C_range = (1.0e-6, 3.5e4)#(1.0e-6, 3.5e4)
gamma_range =(1.0e-6, 32)#(1.0e-6, 32)
r_range = (0, 10)#(-10, 10)
degree_range = (1, 3)
def map_labels(y):
    return ['normal' if label == 'normal' else 'attack' for label in y]
def calc_and_write_data(pre):
    global TP,TN,FP,FN,detection_rate,false_alarm_rate,precision,f1
    # 再分類されたラベル
    y_test_mapped = map_labels(t_end)
    y_pred_mapped = map_labels(pre)

    # 混同行列の計算
    cm = confusion_matrix(y_test_mapped, y_pred_mapped, labels=['normal', 'attack'])

    # 混同行列から各値を抽出
    TN, FP, FN, TP = cm.ravel()
    # 検知率（再現率）
    detection_rate = recall_score(y_test_mapped, y_pred_mapped, pos_label='attack')
    # 誤警報率
    false_alarm_rate = FP / (TN + FP)
    # 適合率
    precision = precision_score(y_test_mapped, y_pred_mapped, pos_label='attack')
    # F値
    f1 = f1_score(y_test_mapped, y_pred_mapped, pos_label='attack')
    # t_end と predictions が pandas.Series の場合に備えて iloc を使用
    incorrect_indices = [i for i in range(len(t_end)) if t_end.iloc[i] != pre[i]]
    # 誤分類されたラベルを取得
    incorrect_labels = [(t_end.iloc[i], pre[i]) for i in incorrect_indices]
    # 誤分類されたラベルを正解ラベルでソート
    sorted_incorrect_labels = sorted(incorrect_labels, key=lambda x: x[0])
    # 誤分類された "正解ラベル" の数をカウント
    true_label_counts = pd.Series([true_label for true_label, pred_label in incorrect_labels]).value_counts()
    # 誤分類された "予測ラベル" の数をカウント
    pred_label_counts = pd.Series([pred_label for true_label, pred_label in incorrect_labels]).value_counts()
    # ファイルに結果を書き込む
    with open(output_file, 'a', encoding='utf-8') as f:

        f.write(f"True Positives (TP): {TP}\n")
        f.write(f"True Negatives (TN): {TN}\n")
        f.write(f"False Positives (FP): {FP}\n")
        f.write(f"False Negatives (FN): {FN}\n")

        # 検知率（再現率）
        f.write(f"検知率: {detection_rate:.4f}\n")
        # 誤警報率
        f.write(f"誤警報率: {false_alarm_rate:.4f}\n")
        # 適合率
        f.write(f"適合率: {precision:.4f}\n")
        # F値
        f.write(f"F値: {f1:.4f}\n")
        # ソート後の結果をファイルに書き込み
        f.write("誤分類したデータ (正解ラベルでソート):\n")
        for true_label, pred_label in sorted_incorrect_labels:
            f.write(f"True: {true_label}, Predicted: {pred_label}\n")

        # 結果の表示 (各クラスの誤分類数)
        f.write("\n誤分類(正解):\n")
        f.write(true_label_counts.to_string() + "\n")

        f.write("\n誤分類(予測):\n")
        f.write(pred_label_counts.to_string() + "\n")
def calc_and_write_accuracy(t_data,pre,accuracy,name):
    # t_end と predictions が pandas.Series の場合に備えて iloc を使用
    incorrect_indices = [i for i in range(len(t_data)) if t_data.iloc[i] != pre[i]]
    # 誤分類されたラベルを取得
    incorrect_labels = [(t_data.iloc[i], pre[i]) for i in incorrect_indices]
    # 誤分類された "正解ラベル" の数をカウント
    true_label_counts = pd.Series([true_label for true_label, pred_label in incorrect_labels]).value_counts()
    # 誤分類された "予測ラベル" の数をカウント
    pred_label_counts = pd.Series([pred_label for true_label, pred_label in incorrect_labels]).value_counts()
    # ファイルに結果を書き込む
    with open(output_file, 'a', encoding='utf-8') as f:

        # 結果の表示 (各クラスの誤分類数)
        f.write(f"\n------{name}-----\n")
        f.write(f"精度: {accuracy}\n")
        f.write("誤分類(正解):\n")
        f.write(true_label_counts.to_string() + "\n")
        f.write("\n誤分類(予測):\n")
        f.write(pred_label_counts.to_string() + "\n")
    return accuracy
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
    label_map = {
            'normal.': 'normal',
            'back.': 'DoS', 'land.': 'DoS', 'neptune.': 'DoS', 'pod.': 'DoS', 'smurf.': 'DoS', 'teardrop.': 'DoS',
            'ipsweep.': 'Probe', 'nmap.': 'Probe', 'portsweep.': 'Probe', 'satan.': 'Probe',
            'ftp_write.': 'R2L', 'guess_passwd.': 'R2L', 'imap.': 'R2L', 'multihop.': 'R2L', 'phf.': 'R2L', 'spy.': 'R2L', 'warezclient.': 'R2L', 'warezmaster.': 'R2L',
            'buffer_overflow.': 'U2R', 'loadmodule.': 'U2R', 'perl.': 'U2R', 'rootkit.': 'U2R'
        }
    # ラベルをマッピング
    data_frame['label'] = data_frame['label'].map(label_map)
    df_train = data_frame.sample(frac=0.1, random_state=42)
    df_check = data_frame.sample(frac=0.1, random_state=41)
    df_test = data_frame.sample(frac=0.1, random_state=39)
    x_train = df_train.drop('label', axis=1)
    t_train = df_train['label']
    x_test = df_check.drop('label', axis=1)
    t_test = df_check['label']
    x_end = df_test.drop('label', axis=1)
    t_end = df_test['label']
    return x_train, t_train, x_test, t_test, x_end, t_end
parser = argparse.ArgumentParser(description="説明をここに書く")
parser.add_argument("-o", "--output", help="ファイルの枝番とか")
args = parser.parse_args()
output_file =  "実験データ_"+str(args.output)+".txt"
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
    s_svm_time = time.perf_counter() 
    #print(f"評価中",solution) #デバッグ用    
    if solution[0] == 1:
        svc = svm.SVC(kernel='linear',C=solution[1]*(C_range[1]- C_range[0]) + C_range[0],
                      )
    elif solution[0] == 2:
        svc = svm.SVC(kernel='rbf',  C=solution[1]*(C_range[1]- C_range[0]) + C_range[0],
                      gamma = solution[2]*(gamma_range[1]- gamma_range[0]) + gamma_range[0])
    elif solution[0] == 3:
        svc = svm.SVC(kernel='sigmoid',C = solution[1]*(C_range[1]- C_range[0]) + C_range[0],
                      gamma = solution[2]*(gamma_range[1]- gamma_range[0]) + gamma_range[0], 
                      coef0 = solution[3]*(r_range[1]- r_range[0]) + r_range[0])
    elif solution[0] == 4:
        svc = svm.SVC(kernel='poly',C =  solution[1]*(C_range[1]- C_range[0]) + C_range[0],
                      gamma = solution[2]*(gamma_range[1]- gamma_range[0]) + gamma_range[0], 
                      coef0 = solution[3]*(r_range[1]- r_range[0]) + r_range[0],
                      degree = round(solution[4]*(degree_range[1]- degree_range[0]) + degree_range[0])
                    )
    if flag == 1:
        svc.fit(x_train_std, t_train)#学習セット
        predictions = svc.predict(x_end_std)
        Miss = 1 - accuracy_score(t_end, predictions)
        calc_and_write_data(predictions)
        train_predictions = svc.predict(x_train_std)
        train_accuracy = accuracy_score( t_train, train_predictions)
        ac = calc_and_write_accuracy(t_train,train_predictions,train_accuracy,"学習セット")
        learn_list.append(ac)
        test_predictions = svc.predict(x_test_std)
        test_accuracy = accuracy_score(t_test, test_predictions)
        ac = calc_and_write_accuracy(t_test,test_predictions,test_accuracy,"検証セット")
        test_list.append(ac)
    elif STD == 0:
        svc.fit(x_train_std, t_train)#学習セット
        predictions = svc.predict(x_test_std)#検証セット
        Miss = 1 - accuracy_score(t_test, predictions)
    else: print("標準化をしてください\n")
    return  1/(1+Miss)#ただの評価値
#ABCアルゴリズム
learn_list,test_list =[],[]
TP, TN, FP, FN = 0, 0, 0, 0
detection_rate, false_alarm_rate, precision, f1 = 0, 0, 0, 0
  # 初期化
solution = np.array([2.00000 ,0.01239 ,0.01513 ,0.25119 ,0.34292])
fitness = 0
fitness = evaluate_function(solution,0)
# 解の初期化
#ここにテストセットで分類精度を検証するプログラムを記述（これが最終的な分類精度)
fitness= evaluate_function(solution,1) 
# 結果の出力
print("Solution:", solution)
print("Fitness:", 2 - (1/fitness))
print("Fitness:", fitness)
with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"Best Solution: {str(solution)}\n精度: {str(2 - (1 / fitness))}\n")
        f.write(f"C = {solution[1]*(C_range[1]- C_range[0]) + C_range[0]}\n")
        f.write(f"gamma = {solution[2]*(gamma_range[1]- gamma_range[0]) + gamma_range[0]}\n")
        f.write(f"coef0 = {solution[3]*(r_range[1]- r_range[0]) + r_range[0]}\n")
        f.write(f"d = {solution[4]*(degree_range[1]- degree_range[0]) + degree_range[0]}\n")