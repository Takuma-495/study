import time
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

パラメータcoef0の範囲（実験して良さそうな値出す)
分類精度をちゃんと算出する
初期化の工夫
"""
# SVMのパラメータ範囲を設定
kernels = [1, 2, 3, 4]#[1, 2, 3, 4]
C_range = (1.0e-6, 3.5e4)#(1.0e-6, 3.5e4)
gamma_range =(1.0e-6, 32)#(1.0e-6, 32)
r_range = (0, 10)#(-10, 10)
degree_range = (1, 3) #ここが４，５だと処理終わらなくなる #(1, 3)
svm_iter = int(1.0e7)#100万#制限なし　＝　−１
DEBAG = False #True or False
#ABCのハイパーパラメータ
COLONY_SIZE = 10#コロニーサイズ/2(偶数整数)
LIMIT = 100#偵察バチのパラメータ
CYCLES = 500#サイクル数
DIM = 5# 次元数 (カーネル ,C,γ,r, degree)
#実験回数
EX_CYCLE = 5
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
    x_trai = df_train.drop('label', axis=1)
    t_trai = df_train['label']
    x_ch = df_check.drop('label', axis=1)
    t_ch = df_check['label']
    x_tes = df_test.drop('label', axis=1)
    t_tes = df_test['label']
    return x_trai, t_trai, x_ch, t_ch, x_tes, t_tes

parser = argparse.ArgumentParser(description="説明をここに書く")
parser.add_argument("-s","--std", type=int, default=0, help="0で標準化")
parser.add_argument("-d","--data", type=str,default="kdd99" , help="データセットネーム")
parser.add_argument("-o", "--output", default= 0, help="ファイルの枝番とか")
args = parser.parse_args()
dataset_name = args.data
output_file = args.data+ "_"+str(args.output)+".txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(f"正規化(0で有効): {args.std}\n")
    f.write(f"データセット: {args.data}\n")
    f.write(f"打ち切り学習回数: {format(svm_iter, '.1e')}\n")
    f.write(f'r_range = {r_range}\n')
    f.write(f"コロニーサイズ: {COLONY_SIZE}\n")
    f.write(f"偵察バチのLIMIT: {LIMIT}\n")
    f.write(f"サイクル数: {CYCLES}\n")
    f.write(f"試行回数: {EX_CYCLE}\n")
STD = args.std#0で標準化有
std_scaler = MinMaxScaler()
#std_scaler = StandardScaler()
# データセットのロード
x_train, t_train, x_test, t_test, x_end, t_end = load_kdd99()
DEFAULT_ACCURACY =   0.9978543378810575
# データをトレーニングセットとテストセットに分割する
std_scaler.fit(x_train)  # 訓練データでスケーリングパラメータを学習
x_train_std = std_scaler.transform(x_train)  # 訓練データの標準化
x_test_std = std_scaler.transform(x_test)    # テストデータの標準化
x_end_std = std_scaler.transform(x_end)

# 評価関数
def evaluate_function(solution,flag):
    global svm_time
    global STD
    global eva_count
    s_svm_time = time.perf_counter() 
    print(f"評価中",solution) #デバッグ用
    if solution[0] == 1:
        svc = svm.SVC(kernel='linear',C=solution[1]*(C_range[1]- C_range[0]) + C_range[0],
                      verbose=DEBAG,max_iter= svm_iter)
    elif solution[0] == 2:
        svc = svm.SVC(kernel='rbf',  C=solution[1]*(C_range[1]- C_range[0]) + C_range[0],
                      gamma = solution[2]*(gamma_range[1]- gamma_range[0]) + gamma_range[0],verbose=DEBAG,max_iter= svm_iter)
    elif solution[0] == 3:
        svc = svm.SVC(kernel='sigmoid',C = solution[1]*(C_range[1]- C_range[0]) + C_range[0],
                      gamma = solution[2]*(gamma_range[1]- gamma_range[0]) + gamma_range[0], 
                      coef0 = solution[3]*(r_range[1]- r_range[0]) + r_range[0],verbose=DEBAG,max_iter= svm_iter)
    elif solution[0] == 4:
        svc = svm.SVC(kernel='poly',C =  solution[1]*(C_range[1]- C_range[0]) + C_range[0],
                      gamma = solution[2]*(gamma_range[1]- gamma_range[0]) + gamma_range[0], 
                      coef0 = solution[3]*(r_range[1]- r_range[0]) + r_range[0],
                      degree = round(solution[4]*(degree_range[1]- degree_range[0]) + degree_range[0])
                      ,verbose=DEBAG,max_iter= svm_iter)
    else:
        print("カーネル関数エラー")
    if flag == 1:
        svc.fit(x_train_std, t_train)#学習セット
        predictions = svc.predict(x_end_std)
        accuracy = accuracy_score(t_end, predictions)
        calc_and_write_data(predictions)
        train_predictions = svc.predict(x_train_std)
        train_accuracy = accuracy_score( t_train, train_predictions)
        ac = calc_and_write_accuracy(t_train,train_predictions,train_accuracy,"学習セット")
        learn_list.append(ac)
        calc_and_write_accuracy(t_train,train_predictions,train_accuracy,"学習セット")
        test_predictions = svc.predict(x_test_std)
        test_accuracy = accuracy_score(t_test, test_predictions)
        calc_and_write_accuracy(t_test,test_predictions,test_accuracy,"検証セット")
        ac = calc_and_write_accuracy(t_test,test_predictions,test_accuracy,"検証セット")
        test_list.append(ac)
    elif STD == 0:
        svc.fit(x_train_std, t_train)#学習セット
        predictions = svc.predict(x_test_std)#検証セット
        accuracy = accuracy_score(t_test, predictions)
    else:
        svc.fit(x_train, t_train)
        predictions = svc.predict(x_test)
        accuracy = accuracy_score(t_test, predictions)

    e_svm_time = time.perf_counter()
    svm_time += e_svm_time - s_svm_time
    eva_count = eva_count + 1
    return  1/(2-accuracy) 

def roulette_kernel(new_solution,solutions):
    k = np.random.randint(0, COLONY_SIZE)
    while solutions[i][0] != solutions[k][0] and k == i :
         k = np.random.randint(0, COLONY_SIZE)
    if  np.random.rand() < fitness[k] /(fitness[i] + fitness[k]):
        new_solution[0] = solutions[k][0]
    else :
        trials[i] += 1
    return 
"""
def random_kernel(new_solution): 
    temp = new_solution[j]
    while  new_solution[j] != temp:
          new_solution[j] =np.random.randint(1,5)\
"""
#範囲制限関数
"""
def clip(index,solution):
    if   index == 1:
        return np.clip(solution[1], *C_range)
    if index == 2:
        return np.clip(solution[2], *gamma_range)
    if index == 3:
        return np.clip(solution[3], *r_range)
    if index == 4:
        return np.clip(solution[4], *degree_range)
    return None
"""
# 解の初期化
def initialize_solution():
    kernel = np.random.choice(kernels)
    return [kernel, np.random.rand(), np.random.rand(), np.random.rand(), np.random.rand()]

def bee(func_i, solutions, fitness, trials):
    """解の更新

    """
    global best_fitness
    global best_solution
    new_solution = solutions[func_i].copy()
    j = np.random.randint(0, new_solution[0] + 1)#カーネル関数によって次元が変わる
    if j != 0:
        k = np.random.randint(0, COLONY_SIZE)
        while k == func_i:
            k = np.random.randint(0, COLONY_SIZE)
        new_solution[j] = solutions[func_i][j] + np.random.uniform(-1, 1) * (solutions[func_i][j] - solutions[k][j])
        new_solution[j] = np.clip(new_solution[j],0,1)
        #degreeを丸めた値が更新後も変わらない場合は評価しない
        if j == 4 and round(solutions[func_i][j]*(degree_range[1]- degree_range[0]) + degree_range[0]) ==  round(new_solution[j]*(degree_range[1]- degree_range[0]) + degree_range[0]):
            trials[func_i] += 1
            return
    else:#カーネル関数の更新
        roulette_kernel(new_solution,solutions)
    new_fitness = evaluate_function(new_solution,0)
    if new_fitness > fitness[func_i]:
        solutions[func_i] = new_solution
        fitness[func_i] = new_fitness
        trials[func_i] = 0
        if fitness[i] > best_fitness:
            best_fitness = fitness[i]  # ここは2つの変数を一つにまとめたほうが良いかも
            best_solution = solutions[i]
    else:
        trials[func_i] += 1
#ルーレット選択
def roulette_wheel_selection(fitness):
    total_fitness = np.sum(fitness)
    if total_fitness == 0:
        # フィットネスが全て0の場合、ランダムに選択
        return np.random.randint(len(fitness))
    probabilities = fitness / total_fitness
    cumulative_probabilities = np.cumsum(probabilities)
    r = np.random.rand()
    # np.searchsorted を使って高速にインデックスを取得
    selected_index = np.searchsorted(cumulative_probabilities, r)
    return selected_index
#ABCアルゴリズム
best_box = []#各試行の最良値
All_time = []
TP_list, TN_list, FP_list, FN_list = [], [], [], []
detection_rate_list, false_alarm_rate_list = [], []
precision_list, f1_list = [], []
learn_list,test_list =[],[]
timelist =[]
fig, ax = plt.subplots()
#ax.set_title('Best Fitness over Generations')
#ax.set_xlabel('Generation')
#ax.set_ylabel('Best Fitness')
#ax.grid(True)
for e in range(EX_CYCLE):
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write("###############\n\n")
        f.write(f"{e+1}試行目\n\n")
        f.write("###############\n")
    # 初期化
    best_solution = 0
    best_fitness = 0
    svm_time = 0 #時間測定用
    eva_count =0
    fitness_history = []
     # 各変数を個別に初期化
    TP, TN, FP, FN = 0, 0, 0, 0
    detection_rate, false_alarm_rate, precision, f1 = 0, 0, 0, 0
    solutions = np.zeros((COLONY_SIZE, DIM))
    fitness = np.zeros(COLONY_SIZE)
    trials = np.zeros(COLONY_SIZE)

    # 解の初期化
    s_all_time = time.perf_counter()
    for i in range(COLONY_SIZE):
        solutions[i] = initialize_solution()
        fitness[i] = evaluate_function(solutions[i],0)

        if fitness[i] > best_fitness:
            best_fitness = fitness[i]  # ここは2つの変数を一つにまとめたほうが良いかも
            best_solution = solutions[i]
            trials[i] = 0
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"初期化{i+1} , Fitness:{fitness[i]}\n")
            f.write(f"{solutions[i]}\n")
        print(f"初期化{i} ", "Fitness:", fitness[i])
        print(solutions[i])

    #ABC   
    for _ in range(CYCLES):
        # 働きバチ
        for i in range(COLONY_SIZE):
            bee(i, solutions, fitness, trials)

        # 追従バチ
        sum_fitness = sum(fitness)
        for i in range(COLONY_SIZE):
            if np.random.rand() < fitness[i] / sum_fitness:
                bee(i, solutions, fitness, trials)
          #ルーレット選択
             #selected = roulette_wheel_selection(fitness)
             #bee(selected, solutions, fitness, trials)
        # 偵察バチ
        for i in range(COLONY_SIZE):
            if trials[i] > LIMIT:
                solutions[i] = initialize_solution()
                fitness[i] = evaluate_function(solutions[i],0)
                trials[i] = 0
                if fitness[i] > best_fitness:
                     best_fitness = fitness[i]  # ここは2つの変数を一つにまとめたほうが良いかも
                     best_solution = solutions[i]

        fitness_history.append(2 - (1 / best_fitness))  # 結果表示用配列
        print("Generation:", _ + 1, "Best Fitness:", 2 - (1 / best_fitness))
        print(best_solution)
        # テキストデータをファイルに書き込む
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"Gen: {str(_ + 1)}, Best: {str(2 - (1 / best_fitness))}\n")
            f.write(str(best_solution) + "\n") 
    #ここにテストセットで分類精度を検証するプログラムを記述（これが最終的な分類精度)
    best_fitness= evaluate_function(best_solution,1)   
    e_all_time = time.perf_counter()
    execution_time = e_all_time - s_all_time
    best_box.append(2 - (1 / best_fitness))
    All_time.append(execution_time)
    TP_list.append(TP)
    TN_list.append(TN)
    FP_list.append(FP)
    FN_list.append(FN)
    detection_rate_list.append(detection_rate)
    false_alarm_rate_list.append(false_alarm_rate)
    precision_list.append(precision)
    f1_list.append(f1)
    timelist.append(svm_time)
    # 結果の出力
    print("Best Solution:", best_solution)
    print("Best Fitness:", 2 - (1/best_fitness))
    print("default_Fitness:", DEFAULT_ACCURACY)
    # 実行時間の出力
    print(f"実行時間: {execution_time:.4f}秒")
    print(f"SVMの実行時間: {svm_time:.4f}秒")
    #print(f"デフォルト実行時間: {time:.4f}秒")
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"Best Solution: {str(best_solution)}\n精度: {str(2 - (1 / best_fitness))}\n")
        f.write(f"C = {best_solution[1]*(C_range[1]- C_range[0]) + C_range[0]}\n")
        f.write(f"gamma = {best_solution[2]*(gamma_range[1]- gamma_range[0]) + gamma_range[0]}\n")
        f.write(f"coef0 = {best_solution[3]*(r_range[1]- r_range[0]) + r_range[0]}\n")
        f.write(f"d = {best_solution[4]*(degree_range[1]- degree_range[0]) + degree_range[0]}\n")
        f.write(f"デフォルト精度: {DEFAULT_ACCURACY}\n")
        f.write(f"実行時間: {execution_time:.4f}秒\n")
        f.write(f"SVMの実行時間: {svm_time:.4f}秒\n")
        f.write(f"SVMの実行時間H: {svm_time/3600:.4f}時間\n")
        f.write(f"評価回数: {eva_count}回\n")
    # best_fitness の推移をグラフで描画
    # すべての個体の出力
    for i in range(COLONY_SIZE):
        print(f"精度:{2-(1/fitness[i]):.4f}  {solutions[i]}")
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"精度:{2-(1/fitness[i]):.4f}  {solutions[i]}\n")
    plt.figure()
    plt.ylim(0.998, 1)
    plt.plot(range(1, CYCLES + 1), fitness_history, )
    plt.title('Best Fitness over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Best Accuracy')
    plt.grid(True)
    plt.savefig(f"./{dataset_name}_{str(args.output)}-{e}.pdf", bbox_inches="tight")
with open(output_file, 'a', encoding='utf-8') as f:
    f.write("\n--- 最終結果（平均値）---\n")
    f.write(f"平均分類精度: {sum(best_box)/len(best_box)}\n")
    f.write(f"デフォルト精度: {DEFAULT_ACCURACY}\n")
    f.write(f"平均svm実行時間: {sum(timelist)/len(timelist):.4f}秒\n") 
    f.write(f"平均svm実行時間H: {(sum(timelist)/len(timelist))/3600:.4f}時間\n") 
    f.write(f"平均TP: {np.mean(TP_list):.2f}\n")
    f.write(f"平均TN: {np.mean(TN_list):.2f}\n")
    f.write(f"平均FP: {np.mean(FP_list):.2f}\n")
    f.write(f"平均FN: {np.mean(FN_list):.2f}\n")
    f.write(f"平均検知率: {np.mean(detection_rate_list):.4f}\n")
    f.write(f"平均誤警報率: {np.mean(false_alarm_rate_list):.4f}\n")
    f.write(f"平均適合率: {np.mean(precision_list):.4f}\n")
    f.write(f"平均F値: {np.mean(f1_list):.4f}\n")
    f.write(f"平均学習セット分類精度: {np.mean(learn_list):.4f}\n")
    f.write(f"平均検証セット分類精度: {np.mean(test_list):.4f}\n")
print("\n--- 最終結果（平均値）---")
print(f"平均分類精度: {sum(best_box)/len(best_box)}")
print(f"デフォルト精度: {DEFAULT_ACCURACY}")
print(f"平均実行時間: {sum(timelist)/len(timelist):.4f}秒")
print(f"平均TP: {np.mean(TP_list):.2f}")
print(f"平均TN: {np.mean(TN_list):.2f}")
print(f"平均FP: {np.mean(FP_list):.2f}")
print(f"平均FN: {np.mean(FN_list):.2f}")
print(f"平均検知率: {np.mean(detection_rate_list):.4f}")
print(f"平均誤警報率: {np.mean(false_alarm_rate_list):.4f}")
print(f"平均適合率: {np.mean(precision_list):.4f}")
print(f"平均F値: {np.mean(f1_list):.4f}")
print(f"平均学習セット分類精度: {np.mean(learn_list):.4f}")
print(f"平均検証セット分類精度: {np.mean(test_list):.4f}")