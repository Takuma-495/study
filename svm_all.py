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
import concurrent.futures
import argparse
np.set_printoptions(precision=5,suppress=True,floatmode='maxprec_equal')
"""
課題
カーネル関数によって更新パラメータを制御（済)
degreeの値が変わらなかったらもう一度個体生成(済)
標準化したほうがいいデータとしないほうがいいデータがある
→標準化の有無を指定出来たほうがいい(済)
※irisはしないほうがいい
kdd99はしたほうが良かった
分類精度をちゃんと算出する
多項式カーネルかつ特定のパラメータセットでは計算量が膨大になる(評価困難)
→d(1-3)に変更(済)
初期化の工夫
ルーレット選択の実装(済)
実験を複数回やってそこから統計的な値を出す機能の追加(済)
学習セット、検証セット、テストセットでの分割
パラメータの範囲（実験して良さそうな値出す)
"""
# SVMのパラメータ範囲を設定
kernels = [1, 2, 3, 4]#[1, 2, 3, 4]
C_range = (1.0e-6, 3.5e4)#(1.0e-6, 3.5e4)
gamma_range =(1.0e-6, 32)#(1.0e-6, 32)
r_range = (-10, 10)#(-10, 10)
degree_range = (1, 3) #ここが４，５だと処理終わらなくなる #(1, 3)
svm_time = 0 #時間測定用
svm_iter = int(1.0e7)#制限なし　＝　−１
DEBAG = False #True or False
#ABCのハイパーパラメータ
COLONY_SIZE = 10#コロニーサイズ/2(偶数整数)
LIMIT = 100#偵察バチのパラメータ
CYCLES = 500#サイクル数
DIM = 5# 次元数 (カーネル ,C,γ,r, degree)
#実験回数
ex_cycle = 10
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

parser = argparse.ArgumentParser(description="説明をここに書く")
parser.add_argument("-s","--std", type=int, default=0, help="0で標準化")
parser.add_argument("-d","--data", type=str, required=True, help="データセットネーム")
parser.add_argument("-o", "--output", default= 0, help="ファイルの枝番とか")
args = parser.parse_args()
dataset_name = args.data  # ここを 'iris', 'wine', 'digits', 'breast_cancer' , 'kdd99'のいずれかに変える
output_file = args.data+ "_"+str(args.output)+".txt"
with open(output_file, 'w') as f:
    f.write(f"標準化(0で有効): {args.std}\n")
    f.write(f"データセット: {args.data}\n")
    f.write(f"打ち切り学習回数: {format(svm_iter, '.1e')}\n")
    f.write(f'r_range = {r_range}\n')
    f.write(f"コロニーサイズ: {COLONY_SIZE}\n")
    f.write(f"偵察バチのLIMIT: {LIMIT}\n")
    f.write(f"サイクル数: {CYCLES}\n")
    f.write(f"試行回数: {ex_cycle}\n")
STD = args.std#0で標準化有
std_scaler = StandardScaler()
# データセットのロード
if dataset_name == 'iris':
    dataset = load_iris()
    default_accuracy = 0.9777777777777777
elif dataset_name == 'wine':
    dataset = load_wine()
    default_accuracy = 1.0
elif dataset_name == 'digits':
    dataset = load_digits()
    default_accuracy = 0.9851851851851852#標準化あり
  #  default_accuracy = 0.9907407407407407#標準化なし
elif dataset_name == 'cancer':
    dataset = load_breast_cancer()
    default_accuracy = 0.9766081871345029
elif dataset_name == 'kdd99':
    x_train, t_train, x_test, t_test, x_end, t_end = load_kdd99()
    default_accuracy = 0.9978543378810575
else:
    raise ValueError("Invalid dataset name. Choose from 'iris', 'wine', 'digits', 'cancer'.")

# データをトレーニングセットとテストセットに分割する

if dataset_name != 'kdd99':
    df_dataset = pd.DataFrame(data = dataset.data,columns=dataset.feature_names)
    x_train, x_test, t_train, t_test = train_test_split(
    dataset.data, dataset.target, test_size=0.3, random_state=0)
std_scaler.fit(x_train)  # 訓練データでスケーリングパラメータを学習
x_train_std = std_scaler.transform(x_train)  # 訓練データの標準化
x_test_std = std_scaler.transform(x_test)    # テストデータの標準化
if dataset_name == 'kdd99':
   x_end_std = std_scaler.transform(x_end)

# 評価関数
def evaluate_function(solution,flag):
    global svm_time
    global STD
    s_svm_time = time.perf_counter() 
    #print(f"評価中",solution) #デバッグ用
    if solution[0] == 1:
        svc = svm.SVC(kernel='linear', C = solution[1],verbose=DEBAG,max_iter= svm_iter)
    elif solution[0] == 2:
        svc = svm.SVC(kernel='rbf',  C = solution[1],  gamma = solution[2],verbose=DEBAG,max_iter= svm_iter)
    elif solution[0] == 3:
        svc = svm.SVC(kernel='sigmoid',C = solution[1],  gamma = solution[2], coef0 = solution[3],verbose=DEBAG,max_iter= svm_iter)
    elif solution[0] == 4:
        svc = svm.SVC(kernel='poly',C = solution[1],
                      gamma = solution[2], coef0 = solution[3], degree = round(solution[4]),verbose=DEBAG,max_iter= svm_iter)
    else:
        print("カーネル関数エラー")
    if flag == 1:
         svc.fit(x_train_std, t_train)#学習セット
         predictions = svc.predict(x_end_std)
         accuracy = accuracy_score(t_end, predictions)
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

    return  1/(2-accuracy)

#タイムアウト付き評価関数
def evaluate_function_with_timeout(solution, timeout=5):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(evaluate_function, solution)
        try:
            result = future.result(timeout=timeout)
            return result
        except concurrent.futures.TimeoutError:
            print("処理がタイムアウトしました")
            return None
def roulette_kernel(new_solution,solutions):
    k = np.random.randint(0, COLONY_SIZE)
    while solutions[i][0] != solutions[k][0] and k == i :
         k = np.random.randint(0, COLONY_SIZE)
    if  np.random.rand() < fitness[k] /(fitness[i] + fitness[k]):
        new_solution[0] = solutions[k][0]
    else :
        trials[i] += 1
    return 
def random_kernel(new_solution): 
    temp = new_solution[j]
    while  new_solution[j] != temp:
          new_solution[j] =np.random.randint(1,5)
#範囲制限関数
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

# 解の初期化
def initialize_solution():
    C = np.random.uniform(*C_range)
    kernel = np.random.choice(kernels)
    gamma = np.random.uniform(*gamma_range)
    r = np.random.uniform(*r_range)
    degree = np.random.uniform(*degree_range)
    return [kernel, C, gamma, r, degree]

def bee(i, solutions, fitness, trials):
    new_solution = solutions[i].copy()
    j = np.random.randint(0, new_solution[0] + 1)
    if j != 0:
        k = np.random.randint(0, COLONY_SIZE)
        while k == i:
            k = np.random.randint(0, COLONY_SIZE)
        new_solution[j] = solutions[i][j] + np.random.uniform(-1, 1) * (solutions[i][j] - solutions[k][j])
        new_solution[j] = clip(j, new_solution)
        #degreeを丸めた値が更新後も変わらない場合は評価しない
        if j == 4 and round(solutions[i][j]) == round(new_solution[j]):
            trials[i] += 1
            return
    else:#カーネル関数の更新
        roulette_kernel(new_solution,solutions)
    new_fitness = evaluate_function(new_solution,0)
    if new_fitness > fitness[i]:
        solutions[i] = new_solution
        fitness[i] = new_fitness
        trials[i] = 0
    else:
        trials[i] += 1
# ルーレット選択用関数(作るかも)
#ABCアルゴリズム
best_box = []
All_time = []
for e in range(ex_cycle):
    with open(output_file, 'a') as f:
        f.write("###############\n\n")
        f.write(f"{e+1}試行目\n\n")
        f.write("###############\n")
    # 初期化
    best_solution = 0
    best_fitness = 0
    fitness_history = []

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
        with open(output_file, 'a') as f:
            f.write(f"初期化{i+1} , Fitness:{fitness[i]}\n")
            f.write(f"{solutions[i]}\n")
        print(f"初期化{i} ", "Fitness:", fitness[i])
        print(solutions[i])
    
    # ABC   
    for _ in range(CYCLES):
        # 働きバチ
        for i in range(COLONY_SIZE):
            bee(i, solutions, fitness, trials)

        # 追従バチ
        sum_fitness = sum(fitness)
        for i in range(COLONY_SIZE):
            if np.random.rand() < fitness[i] / sum_fitness:
                bee(i, solutions, fitness, trials)

        # 偵察バチ
        for i in range(COLONY_SIZE):
            if trials[i] > LIMIT:
                solutions[i] = [
                    np.random.choice(kernels),
                    C_range[0] + C_range[1] - solutions[i][1],
                    gamma_range[0] + gamma_range[1] - solutions[i][2],
                    r_range[0] + r_range[1] - solutions[i][3],
                    degree_range[0] + degree_range[1] - solutions[i][4]
                ]
                fitness[i] = evaluate_function(solutions[i],0)
                trials[i] = 0
        
        best_fitness = np.max(fitness)  # ここは2つの変数を一つにまとめたほうが良いかも
        fitness_history.append(2 - (1 / best_fitness))  # 結果表示用配列
        max_index = np.where(fitness == best_fitness)[0][0]
        best_solution = solutions[max_index]
        #ここにテストセットで分類精度を検証するプログラムを記述（これが最終的な分類精度)
        if dataset_name == 'kdd99':
           best_fitness= evaluate_function(best_solution,1)
        print("Generation:", _ + 1, "Best Fitness:", 2 - (1 / best_fitness))
        print(best_solution)

        # テキストデータをファイルに書き込む
        with open(output_file, 'a') as f:
            f.write(f"Gen: {str(_ + 1)}, Best: {str(2 - (1 / best_fitness))}\n")
            f.write(str(best_solution) + "\n")
    
    e_all_time = time.perf_counter()
    execution_time = e_all_time - s_all_time
    best_box.append(2 - (1 / best_fitness))
    All_time.append(execution_time)
    # 結果の出力
    print("Best Solution:", best_solution)
    print("Best Fitness:", 2 - (1/best_fitness))
    print("default_Fitness:", default_accuracy)
    # 実行時間の出力
    print(f"実行時間: {execution_time:.4f}秒")
    print(f"SVMの実行時間: {svm_time:.4f}秒")
    #print(f"デフォルト実行時間: {time:.4f}秒")
    with open(output_file, 'a') as f:
        f.write(f"Best Solution: {str(best_solution)}\nBest Fitness: {str(2 - (1 / best_fitness))}\n")
        f.write(f"default Fitness: {default_accuracy}\n")
        f.write(f"実行時間: {execution_time:.4f}秒\n")
        f.write(f"SVMの実行時間: {svm_time:.4f}秒\n")
    # best_fitness の推移をグラフで描画
    plt.figure()
    plt.plot(range(1, CYCLES + 1), fitness_history, marker='o')
    plt.title('Best Fitness over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.grid(True)
    #plt.show()
    plt.savefig(f"./{dataset_name}_{str(args.output)}-{e}.pdf", bbox_inches="tight")
    # すべての個体の出力
    for i in range(COLONY_SIZE):
        print(f"評価値:{2-(1/fitness[i]):.4f}  {solutions[i]}")
        with open(output_file, 'a') as f:
         f.write(f"評価値:{2-(1/fitness[i]):.4f}  {solutions[i]}\n")
with open(output_file, 'a') as f:
    f.write(f"Best Fitness mean: {sum(best_box)/len(best_box)}\n")
    f.write(f"default Fitness: {default_accuracy}\n")
    f.write(f"平均実行時間: {sum(All_time)/len(All_time):.4f}秒\n")      
print(f"Best Fitness mean: {sum(best_box)/len(best_box)}\n")
print(f"default Fitness: {default_accuracy}\n")
print(f"平均実行時間: {sum(All_time)/len(All_time):.4f}秒\n")

#デフォルトsvm
"""
s = time.perf_counter()
default_svc = svm.SVC()
if STD == 0:
    default_svc.fit(x_train_std, t_train)
    default_predictions = default_svc.predict(x_test_std)
else:
    default_svc.fit(x_train, t_train)
    default_predictions = default_svc.predict(x_test)
default_accuracy = accuracy_score(t_test, default_predictions)
e = time.perf_counter()
time = e - s
"""
