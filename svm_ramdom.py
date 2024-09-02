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
初期化の工夫
ルーレット選択の実装
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
parser = argparse.ArgumentParser(description="説明をここに書く")
parser.add_argument("-s","--std", type=int, default=0, help="0で標準化")
parser.add_argument("-d","--data", type=str, required=True, help="データセットネーム")
parser.add_argument("-o", "--output", default= 0, help="ファイルの枝番とか")
args = parser.parse_args()
dataset_name = args.data  # ここを 'iris', 'wine', 'digits', 'breast_cancer' , 'kdd99'のいずれかに変える
output_file = args.data+ "_"+str(args.output)+".txt"
with open(output_file, 'w') as f:
        f.write(" \n")
STD = args.std#0で標準化有
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
    x_train, t_train, x_test, t_test = load_kdd99()
    default_accuracy = 0.9978543378810575
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
# SVMのパラメータ範囲を設定
kernels = [1, 2, 3, 4]#[1, 2, 3, 4]
C_range = (1.0e-6, 3.5e4)#(1.0e-6, 3.5e4)
gamma_range =(1.0e-6, 32)#(1.0e-6, 32)
r_range = (-10, 10)#(-10, 10)
degree_range = (2, 5) #ここが４，５だと処理終わらなくなる #(2, 5)
svm_time = 0
svm_iter = 1000#制限なし　＝　−１
DEBAG = False #True or False
#ABCのハイパーパラメータ
COLONY_SIZE = 20#コロニーサイズ/2(偶数整数)
LIMIT = 100#偵察バチのパラメータ
CYCLES = 500#サイクル数
DIM = 5# 次元数 (カーネル ,C,γ,r, degree)

# 評価関数
def evaluate_function(solution):
    global svm_time
    global STD
    s_svm_time = time.perf_counter() 
   # print(f"評価中",solution) #デバッグ用
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
    
    if STD == 0:
        svc.fit(x_train_std, t_train)
        predictions = svc.predict(x_test_std)
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
#適応度関数
# def fit(x):
# return 1/(1+x)

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

# ルーレット選択用関数(作るかも)

#ABCアルゴリズム
#初期化
best_solution = 0
best_fitness = 0
fitness_history = []

solutions = np.zeros((COLONY_SIZE,DIM))
fitness = np.zeros(COLONY_SIZE)
trials = np.zeros(COLONY_SIZE)
#解の初期化
s_all_time = time.perf_counter()
for i in range(COLONY_SIZE):
    solutions[i] = initialize_solution()
    fitness[i] = evaluate_function(solutions[i])
    if(fitness[i] > best_fitness):
        best_fitness = fitness[i] #ここは2つの変数を一つにまとめたほうが良いかも
        best_solution = solutions[i]
        trials[i] = 0
    print(f"初期化{i} ", "Fitness:",fitness[i])
    print( solutions[i])
for _ in range(CYCLES):
  # 働きバチ
    for i in range(COLONY_SIZE):
        new_solution = solutions[i].copy()
        j = np.random.randint(0, new_solution[0]+1)
        if(j !=0 ):
            k = np.random.randint(0, COLONY_SIZE)
            while k == i:#(k≠i)
                k = np.random.randint(0, COLONY_SIZE)
            new_solution[j] = solutions[i][j] +np.random.uniform(-1, 1) * (solutions[i][j] - solutions[k][j])
            new_solution[j] = clip(j, new_solution)
            if(j == 4 and round( solutions[i][j]) == round( new_solution[j])):#整数の場合変化しなかったら更新しない
                i+=1
                continue
      #以下はカーネル関数の選択方法（研究の要)(いったんランダムで)(どうせなら変えてみて評価したいと思ったけど評価しない方が無駄がないからルーレット選択もあり)
        else:
            temp = new_solution[j]
            while  new_solution[j] != temp:
                 new_solution[j] =np.random.randint(1,5)
        new_fitness = evaluate_function(new_solution)
      #  if new_fitness is None:  # タイムアウトの場合は次に進む
     #       continue
        if new_fitness > fitness[i]:
            solutions[i] = new_solution
            fitness[i] = new_fitness
            trials[i] = 0
        else:
            trials[i] += 1
  # 追従バチ
    sum_fitness = 0
    for i in range(COLONY_SIZE):
        sum_fitness += fitness[i]
    for i in range(COLONY_SIZE):
        if np.random.rand() < fitness[i] / sum_fitness :
            new_solution = solutions[i].copy()
            j = np.random.randint(0, new_solution[0]+1)
            if(j !=0 ):
                k = np.random.randint(0, COLONY_SIZE)
                while k == i:#(k≠i)
                    k = np.random.randint(0, COLONY_SIZE)
                new_solution[j] = solutions[i][j] +np.random.uniform(-1, 1) * (solutions[i][j] - solutions[k][j])
                new_solution[j] = clip(j, new_solution)
                if(j == 4 & round( solutions[i][j]) ==round( new_solution[j])):
                    i+=1
                    continue
#以下はカーネル関数の選択方法（研究の要)(いったんランダムで)
            else:
               temp = new_solution[j]
               while  new_solution[j] != temp:
                   new_solution[j] =np.random.randint(1,5)
               new_fitness = evaluate_function(new_solution)
       # if new_fitness is None:  # タイムアウトの場合は次に進む
       #     continue
        if new_fitness > fitness[i]:
            solutions[i] = new_solution
            fitness[i] = new_fitness
            trials[i] = 0
        else:
            trials[i] += 1
  # 偵察バチ
    for i in range(COLONY_SIZE):
        if trials[i] > LIMIT:
            solutions[i] = [np.random.choice(kernels),
                            C_range[0] + C_range[1] - solutions[i][1],
                            C_range[0] + C_range[1] - solutions[i][1],
                            gamma_range[0] + gamma_range[1] - solutions[i][2],
                            r_range[0] + r_range[1] - solutions[i][3],
                            degree_range[0] + degree_range[1] - solutions[i][4]
                            ]
            fitness[i] = evaluate_function(solutions[i])
       # if new_fitness is None:  # タイムアウトの場合は次に進む
        #    continue
        trials[i] = 0
    best_fitness = np.max(fitness) #ここは2つの変数を一つにまとめたほうが良いかも
    fitness_history.append(2 - (1/best_fitness))#結果表示用配列
    max_index = np.where(fitness == best_fitness)[0][0]
    best_solution = solutions[max_index]
    print("Generation:", _ + 1, "Best Fitness:",2 - (1/best_fitness))
    print(best_solution)
    with open(output_file, 'a') as f:
        f.write(f"Gen: {str(_ + 1)}, Best: {str(2 - (1 / best_fitness))}\n")
        f.write(str(best_solution) + "\n")
e_all_time = time.perf_counter()
execution_time = e_all_time - s_all_time

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
plt.savefig(f"./{dataset_name}.pdf", bbox_inches="tight")

# すべての個体の出力
for i in range(COLONY_SIZE):
    print(f"評価値:{2-(1/fitness[i]):.4f}  {solutions[i]}\n")