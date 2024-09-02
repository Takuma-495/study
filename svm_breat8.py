from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_wine
from sklearn.datasets import load_digits
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
np.set_printoptions(precision=5,suppress=True,floatmode='maxprec_equal')
'''
課題
カーネル関数によって更新パラメータを制御（済)
degreeの値が変わらなかったらもう一度個体生成
'''
dataset_name = 'breast_cancer'  # ここを 'iris', 'wine', 'digits', 'breast_cancer' のいずれかに変える

# データセットのロード
if dataset_name == 'iris':
    dataset = load_iris()
elif dataset_name == 'wine':
    dataset = load_wine()
elif dataset_name == 'digits':
   dataset = load_digits()
elif dataset_name == 'breast_cancer':
   dataset = load_breast_cancer()
else:
    raise ValueError("Invalid dataset name. Choose from 'iris', 'wine', 'digits', 'breast_cancer'.")

# データをトレーニングセットとテストセットに分割する
x_train, x_test, t_train, t_test = train_test_split(
    dataset.data, dataset.target, test_size=0.3, random_state=0)
df_dataset = pd.DataFrame(data=dataset.data,columns=dataset.feature_names)
# 標準化
std_scaler = StandardScaler()
std_scaler.fit(x_train)  # 訓練データでスケーリングパラメータを学習
x_train_std = std_scaler.transform(x_train)  # 訓練データの標準化
x_test_std = std_scaler.transform(x_test)    # テストデータの標準化
# SVMのパラメータ範囲を設定
kernels = [1, 2, 3]#[1, 2, 3, 4]
C_range = (1.0e-1, 3.5e1)#(1.0e-6, 3.5e4)
gamma_range =(1.0e-1, 32)#(1.0e-6, 32)
r_range = (-10, 10)
degree_range = (2, 5)

#ABCのハイパーパラメータ
colony_size = 20#コロニーサイズ(偶数整数)
limit = 100#偵察バチのパラメータ
cycles = 50000#サイクル数
dim = 5# 次元数 (カーネル ,C,γ,r, degree)

# 評価関数
def evaluate_function(solution):
    if solution[0] == 1:
         svc = svm.SVC(kernel='linear', C = solution[1])
    elif solution[0] == 2:
         svc = svm.SVC(kernel='rbf',  C = solution[1],  gamma = solution[2])
    elif solution[0] == 3:
        svc = svm.SVC(kernel='sigmoid',C = solution[1],  gamma = solution[2], coef0 = solution[3] )
    elif solution[0] == 4:
        svc = svm.SVC(kernel='poly',C = solution[1],  gamma = solution[2], coef0 = solution[3], degree = round(solution[4]))
    else:
        print("カーネル関数エラー")
   # print('fitはじめ')
   # print(solution)
    svc.fit(x_train_std, t_train)
    predictions = svc.predict(x_test_std)
    accuracy = accuracy_score(t_test, predictions)
    return  1/(2-accuracy)
#適応度関数
# def fit(x):
# return 1/(1+x)
#範囲制限関数
def clip(index,solution):
  if   index == 1:
       return np.clip(solution[1], *C_range)
  elif index == 2:
     return np.clip(solution[2], *gamma_range)
  elif index == 3:
     return np.clip(solution[3], *r_range)
  elif index == 4:
     return np.clip(solution[4], *degree_range)
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

solutions = np.zeros((colony_size,dim))
fitness = np.zeros(colony_size)
trials = np.zeros(colony_size)
#解の初期化
start_time = time.perf_counter()
for i in range(colony_size):
    solutions[i] = initialize_solution()
    fitness[i] = evaluate_function(solutions[i])
    if(fitness[i] > best_fitness):
        best_fitness = fitness[i] #ここは2つの変数を一つにまとめたほうが良いかも
        best_solution = solutions[i]
        trials[i] = 0
    print(f"初期化{i} ", "Best Fitness:",fitness[i])
    print( solutions[i])
for _ in range(cycles):
  # 働きバチ
  for i in range(colony_size // 2):
    new_solution = solutions[i].copy()
    j = np.random.randint(0, new_solution[0]+1)
    if(j !=0 ):
      k = np.random.randint(0, colony_size)
      while k == i:#(k≠i)
        k = np.random.randint(0, colony_size)
      new_solution[j] = solutions[i][j] +np.random.uniform(-1, 1) * (solutions[i][j] - solutions[k][j])
      new_solution[j] = clip(j, new_solution)
      if(j == 4 & round( solutions[i][j]) ==round( new_solution[j])):
        i+=1
        continue
      #以下はカーネル関数の選択方法（研究の要)(いったんランダムで)
    else: new_solution[j] =np.random.randint(1,3)
    new_fitness = evaluate_function(new_solution)
    if new_fitness > fitness[i]:
      solutions[i] = new_solution
      fitness[i] = new_fitness
      trials[i] = 0
    else:   trials[i]+=1
  # 追従バチ
  sum_fitness = 0
  for i in range(colony_size):
    sum_fitness += fitness[i]
  for i in range(colony_size // 2):
    if np.random.rand() < fitness[i] / sum_fitness :
       new_solution = solutions[i].copy()
       j = np.random.randint(0, new_solution[0]+1)
       if(j !=0 ):
          k = np.random.randint(0, colony_size)
          while k == i:#(k≠i)
            k = np.random.randint(0, colony_size)
          new_solution[j] = solutions[i][j] +np.random.uniform(-1, 1) * (solutions[i][j] - solutions[k][j])
          new_solution[j] = clip(j, new_solution)
          if(j == 4 & round( solutions[i][j]) ==round( new_solution[j])):
             i+=1
             continue
#以下はカーネル関数の選択方法（研究の要)(いったんランダムで)
       else: new_solution[j] =np.random.randint(1,3)
       new_fitness = evaluate_function(new_solution)
    if new_fitness > fitness[i]:
      solutions[i] = new_solution
      fitness[i] = new_fitness
      trials[i] = 0
    else: trials[i]+=1
  # 偵察バチ
  for i in range(colony_size):
    if trials[i] > limit:
      solutions[i] = initialize_solution()
      fitness[i] = evaluate_function(solutions[i])
  best_fitness = np.max(fitness) #ここは2つの変数を一つにまとめたほうが良いかも
  fitness_history.append(best_fitness)
  max_index = np.where(fitness == best_fitness)[0][0]
  best_solution = solutions[max_index]
  print("Generation:", _ + 1, "Best Fitness:",2 - (1/best_fitness))
  print( best_solution)
end_time = time.perf_counter()
execution_time = end_time - start_time

# 結果の出力
print("Best Solution:", best_solution)
print("Best Fitness:", 2 - (1/best_fitness))
# 実行時間の出力
print(f"実行時間: {execution_time:.4f}秒")
# best_fitness の推移をグラフで描画
plt.figure()
plt.plot(range(1, cycles + 1), fitness_history, marker='o')
plt.title('Best Fitness over Generations')
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.grid(True)
# plt.show()
plt.savefig(f"./{dataset_name}.pdf", bbox_inches="tight")


# すべての個体の出力
for i in range(colony_size):
    print(f"評価値:{fitness[i]:.4f}  {solutions[i]}")