
import pandas as pd
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


# データを読み込み
x_train, t_train, x_check, t_check, x_test, t_test = load_kdd99()

# 出力部分の追加
# データの形状を出力
print(f"x_train shape: {x_train.shape}")
print(f"t_train shape: {t_train.shape}")
print(f"x_check shape: {x_check.shape}")
print(f"t_check shape: {t_check.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"t_test shape: {t_test.shape}")

# 学習データの先頭部分を出力
print("\n--- Training Data Sample ---")
print(x_train.head())
print(t_train.head())

# 検証データの先頭部分を出力
print("\n--- Validation Data Sample ---")
print(x_check.head())
print(t_check.head())

# テストデータの先頭部分を出力
print("\n--- Test Data Sample ---")
print(x_test.head())
print(t_test)
