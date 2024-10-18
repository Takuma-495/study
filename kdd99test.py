
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
    # 学習セット、検証セット、テストセットの抽出
    df_train = df.sample(frac=0.1, random_state=42)
    df_val = df.sample(frac=0.1, random_state=41)
    df_test = df.sample(frac=0.1, random_state=39)
    x_train = df_train.drop('label', axis=1)  # 特徴データ
    t_train = df_train['label']               # ラベル

    x_check = df_val.drop('label', axis=1)
    t_check = df_val['label']

    x_test = df_test.drop('label', axis=1)
    t_test = df_test['label']
    return x_train, t_train, x_check, t_check, x_test, t_test


# データを読み込み
x_train, t_train, x_check, t_check, x_test, t_test = load_kdd99()

# クラスごとのデータ数を表示する関数
def print_class_distribution(t_train, t_check, t_test):
    print("Training set class distribution:")
    print(t_train.value_counts())
    print("\nValidation set class distribution:")
    print(t_check.value_counts())
    print("\nTest set class distribution:")
    print(t_test.value_counts())

# クラスごとのデータ数を表示
print_class_distribution(t_train, t_check, t_test)
