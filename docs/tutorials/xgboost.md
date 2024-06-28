
# XGBoost

[**こちらの Colab ノートブックでお試しください →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Credit_Scorecards_with_XGBoost_and_W&B.ipynb)

このノートブックでは、XGBoost モデルをトレーニングして提出されたローン申請がデフォルトするかどうかを分類します。XGBoost などのブースティングアルゴリズムを使用することで、ローン評価のパフォーマンスを向上させ、内部のリスク管理機能や外部の規制当局に対しても解釈可能性を維持します。

このノートブックは、Nvidia GTC21 で ScotiaBank の Paul Edwards が行った講演に基づいており、XGBoost を使用して解釈可能性を維持しながらよりパフォーマンスの高いクレジットスコアカードを構築する方法を[紹介](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31327/)しました。また、Scotiabank の Stephen Denton (stephen.denton@scotiabank.com) によって公開された[サンプルコード](https://github.com/rapidsai-community/showcase/tree/main/event_notebooks/GTC_2021/credit_scorecard)も提供され、このノートブック全体で使用します。

### [こちらをクリック](https://wandb.ai/morgan/credit_scorecard)して、このノートブックで構築されたライブの W&B ダッシュボードを表示および操作してください

# このノートブックで

この Colab では、Weights & Biases がどのようにして規制対象の企業をサポートするかを紹介します：
- **データ ETL パイプライン**をトラッキングおよびバージョン管理（ローカルまたは S3 や GCS のようなクラウドサービスで）
- **実験結果**をトラッキングし、トレーニング済みモデルを保存
- **複数の評価メトリクス**を視覚的に検査
- ハイパーパラメーター探索で**パフォーマンスを最適化**

**実験と結果をトラック**

すべてのトレーニングハイパーパラメーターと出力メトリクスをトラックし、Experiments ダッシュボードを生成します：

![credit_scorecard](/images/tutorials/credit_scorecard/credit_scorecard.png)

**ベストなハイパーパラメーターを見つけるためのハイパーパラメーター探索を実行**

Weights & Biases では、[Sweeps 機能](https://docs.wandb.ai/guides/sweeps)や [Ray Tune インテグレーション](https://docs.wandb.ai/guides/sweeps/advanced-sweeps/ray-tune)を使用してハイパーパラメーター探索を行うことができます。詳細なハイパーパラメーター探索オプションの使用方法については、ドキュメントをご覧ください。

![credit_scorecard_2](/images/tutorials/credit_scorecard/credit_scorecard_2.png)

# セットアップ

```bash
!pip install -qq wandb>=0.13.10 dill
!pip install -qq xgboost>=1.7.4 scikit-learn>=1.2.1
```

```python
import ast
import sys
import json
from pathlib import Path
from dill.source import getsource
from dill import detect

import pandas as pd
import numpy as np
import plotly
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp
from sklearn import metrics
from sklearn import model_selection
import xgboost as xgb

pd.set_option("display.max_columns", None)
```

# データ

## AWS S3、Google Cloud Storage および W&B Artifacts

![credit_scorecard_3](/images/tutorials/credit_scorecard/credit_scorecard_3.png)

Weights & Biases **Artifacts** を使用すると、エンドツーエンドのトレーニングパイプラインをログに記録して実験を常に再現可能にすることができます。

データのプライバシーは Weights & Biases にとって重要であり、AWS S3 や Google Cloud Storage などのプライベートクラウドからアーティファクトを作成することをサポートしています。ローカルまたはオンプレミスの W&B をリクエストに応じて提供可能です。

デフォルトでは、W&B はアメリカに位置するプライベートな Google Cloud Storage バケットにアーティファクトファイルを保存します。すべてのファイルは保存時および送信中に暗号化されます。機密ファイルの場合、プライベートな W&B インストールの使用または参照アーティファクトの使用を推奨します。

## アーティファクト参照例
**S3/GCS メタデータを含むアーティファクトの作成**

アーティファクトは S3/GCS オブジェクトの ETag、サイズ、およびバージョン ID（バケットでオブジェクトバージョン管理が有効になっている場合）のようなメタデータのみで構成されます。

```python
run = wandb.init()
artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("s3://my-bucket/datasets/mnist")
run.log_artifact(artifact)
```

**必要なときにアーティファクトをローカルにダウンロード**

W&B はアーティファクトがログに記録されたときに記録されたメタデータを使用して、基になるバケットからファイルを取得します。

```python
artifact = run.use_artifact("mnist:latest", type="dataset")
artifact_dir = artifact.download()
```

参照によるアーティファクトの使用方法、クレデンシャル設定などの詳細については [Artifact References](https://docs.wandb.ai/guides/artifacts/references) をご覧ください。

## W&B にログイン
Weights & Biases にログインします

```python
import wandb

wandb.login()

WANDB_PROJECT = "vehicle_loan_default"
```

## 車両ローンデータセット

L&T の [Vehicle Loan Default Prediction dataset](https://www.kaggle.com/sneharshinde/ltfs-av-data) の簡略化バージョンを使用します。このデータセットは W&B Artifacts に保存されています。

```python
# データを保存するフォルダを指定、存在しない場合は新しいフォルダが作成されます
data_dir = Path(".")
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

id_vars = ["UniqueID"]
targ_var = "loan_default"
```

ピクル関数を作成する関数

```python
def function_to_string(fn):
    return getsource(detect.code(fn))
```

#### W&B Artifacts からデータをダウンロード

W&B Artifacts からデータセットをダウンロードします。最初に W&B の run オブジェクトを作成し、データをダウンロードします。データがダウンロードされたら、ワンホットエンコードされます。この処理されたデータは新しいアーティファクトとして同じ W&B にログされます。データをダウンロードした W&B にログすることで、この新しいアーティファクトを生のデータセットアーティファクトにリンクします。

```python
run = wandb.init(project=WANDB_PROJECT, job_type="preprocess-data")
```

W&B から車両ローンデフォルトデータのサブセットをダウンロードします。これには `train.csv` と `val.csv` ファイルおよびいくつかのユーティルファイルが含まれます。

```python
ARTIFACT_PATH = "morgan/credit_scorecard/vehicle_loan_defaults:latest"
dataset_art = run.use_artifact(ARTIFACT_PATH, type="dataset")
dataset_dir = dataset_art.download(data_dir)
```

```python
from data_utils import (
    describe_data_g_targ,
    one_hot_encode_data,
    load_training_data,
)
```

#### データをワンホットエンコード

```python
# データをデータフレームに読み込む
dataset = pd.read_csv(data_dir / "vehicle_loans_subset.csv")

# データをワンホットエンコード
dataset, p_vars = one_hot_encode_data(dataset, id_vars, targ_var)

# 前処理されたデータを保存
processed_data_path = data_dir / "proc_ds.csv"
dataset.to_csv(processed_data_path, index=False)
```

#### 処理されたデータを W&B Artifacts にログ

```python
# プロセスデータ用の新しいアーティファクトを作成し、作成者の関数を含めて Artifacts に入れる
processed_ds_art = wandb.Artifact(
    name="vehicle_defaults_processed",
    type="processed_dataset",
    description="ワンホットエンコードされたデータセット",
    metadata={"preprocessing_fn": function_to_string(one_hot_encode_data)},
)

# 処理されたデータをアーティファクトに添付
processed_ds_art.add_file(processed_data_path)

# 現在の wandb run にこのアーティファクトをログ
run.log_artifact(processed_ds_art)

run.finish()
```

## トレイン/バリデーション分割の取得

ここでは、wandb run オブジェクトを作成するための代替パターンを示します。以下のセルでは、データセットを分割するコードを `wandb.init() as run` の呼び出しでラップしています。

ここでは：

- wandb run を開始
- Artifacts からワンホットエンコードされたデータセットをダウンロード
- 分割に使用したパラメータをログ 
- 新しい `trndat` および `valdat` データセットを Artifacts にログ
- 自動的に wandb run を終了

```python
with wandb.init(
    project=WANDB_PROJECT, job_type="train-val-split"
) as run:  # config is optional here
    # W&B から車両ローンデフォルトデータのサブセットをダウンロード
    dataset_art = run.use_artifact(
        "vehicle_defaults_processed:latest", type="processed_dataset"
    )
    dataset_dir = dataset_art.download(data_dir)
    dataset = pd.read_csv(processed_data_path)

    # スプリットパラメータを設定
    test_size = 0.25
    random_state = 42

    # スプリットパラメータをログ
    run.config.update({"test_size": test_size, "random_state": random_state})

    # トレイン/バリデーション分割を実行
    trndat, valdat = model_selection.train_test_split(
        dataset,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset[[targ_var]],
    )

    print(f"トレインデータセットサイズ: {trndat[targ_var].value_counts()} \n")
    print(f"バリデーションデータセットサイズL {valdat[targ_var].value_counts()}")

    # 分割データセットを保存
    train_path = data_dir / "train.csv"
    val_path = data_dir / "val.csv"
    trndat.to_csv(train_path, index=False)
    valdat.to_csv(val_path, index=False)

    # プロセスデータ用の新しいアーティファクトを作成し、作成者の関数を含めて Artifacts に入れる
    split_ds_art = wandb.Artifact(
        name="vehicle_defaults_split",
        type="train-val-dataset",
        description="トレインとバリデーションに分割されたプロセスデータセット",
        metadata={"test_size": test_size, "random_state": random_state},
    )

    # 処理されたデータをアーティファクトに添付
    split_ds_art.add_file(train_path)
    split_ds_art.add_file(val_path)

    # アーティファクトをログ
    run.log_artifact(split_ds_art)
```

#### トレーニングデータセットの検査
トレーニングデータセットの概要を取得

```python
trndict = describe_data_g_targ(trndat, targ_var)
trndat.head()
```

### W&B Tables でデータセットをログ

W&B Tables を使用すると、画像、ビデオ、オーディオなどのリッチメディアを含む表形式のデータをログ、クエリ、分析できます。これにより、データセットを理解し、モデルの予測を視覚化し、洞察を共有することができます。詳細については [W&B Tables ガイド](https://docs.wandb.ai/guides/tables) をご覧ください。

```python
# "log-dataset" ジョブタイプで wandb run を作成し、整理を保つ (省略可能)
run = wandb.init(
    project=WANDB_PROJECT, job_type="log-dataset"
)

# データセットのランダム1000行を探索用に W&B Table にログ
table = wandb.Table(dataframe=trndat.sample(1000))

# データセットを W&B ワークスペースにログ
wandb.log({"processed_dataset": table})

# wandb run を終了
wandb.finish()
```

# モデリング

## XGBoost モデルのフィッティング

次に、車両ローン申請がデフォルトするかどうかを分類するために XGBoost モデルをフィッティングします。

### GPU でのトレーニング
XGBoost モデルを GPU でトレーニングしたい場合は、以下のように XGBoost に渡すパラメータに設定を変更してください：

```python
"tree_method": "gpu_hist"
```

#### 1) W&B Run の初期化

```python
run = wandb.init(project=WANDB_PROJECT, job_type="train-model")
```

#### 2) モデルパラメータの設定とログ

```python
base_rate = round(trndict["base_rate"], 6)
early_stopping_rounds = 40
```

```python
bst_params = {
    "objective": "binary:logistic",
    "base_score": base_rate,
    "gamma": 1,  ## def: 0
    "learning_rate": 0.1,  ## def: 0.1
    "max_depth": 3,
    "min_child_weight": 100,  ## def: 1
    "n_estimators": 25,
    "nthread": 24,
    "random_state": 42,
    "reg_alpha": 0,
    "reg_lambda": 0,  ## def: 1
    "eval_metric": ["auc", "logloss"],
    "tree_method": "hist",  # use `gpu_hist` to train on GPU
}
```

XGBoost トレーニングパラメータを W&B run config にログ

```python
run.config.update(dict(bst_params))
run.config.update({"early_stopping_rounds": early_stopping_rounds})
```

#### 3) W&B Artifacts からトレーニングデータをロード

```python
# Artifacts からトレーニングデータをロード
trndat, valdat = load_training_data(
    run=run, data_dir=data_dir, artifact_name="vehicle_defaults_split:latest"
)

## 目標列をシリーズとして抽出
y_trn = trndat.loc[:, targ_var].astype(int)
y_val = valdat.loc[:, targ_var].astype(int)
```

#### 4) モデルのフィッティング、結果の W&B へのログ、およびモデルの W&B Artifacts への保存

すべての XGBoost モデルパラメータをログするために `WandbCallback` を使用します。詳細は [W&B docs](https://docs.wandb.ai/guides/integrations) をご覧ください。他のライブラリ（LightGBM など）も W&B と統合されています。

```python
from wandb.integration.xgboost import WandbCallback

# WandbCallback を使用して XGBoostClassifier を初期化
xgbmodel = xgb.XGBClassifier(
    **bst_params,
    callbacks=[WandbCallback(log_model=True)],
    early_stopping_rounds=run.config["early_stopping_rounds"]
)

# モデルをトレーニング
xgbmodel.fit(trndat[p_vars], y_trn, eval_set=[(valdat[p_vars], y_val)])
```

#### 5) 追加のトレーニングおよび評価メトリクスを W&B にログ

```python
bstr = xgbmodel.get_booster()

# トレーニングおよびバリデーションの予測を取得
trnYpreds = xgbmodel.predict_proba(trndat[p_vars])[:, 1]
valYpreds = xgbmodel.predict_proba(valdat[p_vars])[:, 1]

# 追加のトレーニングメトリクスをログ
false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(
    y_trn, trnYpreds
)
run.summary["train_ks_stat"] = max(true_positive_rate - false_positive_rate)
run.summary["train_auc"] = metrics.auc(false_positive_rate, true_positive_rate)
run.summary["train_log_loss"] = -(
    y_trn * np.log(trnYpreds) + (1 - y_trn) * np.log(1 - trnYpreds)
).sum() / len(y_trn)

# 追加のバリデーションメトリクスをログ
ks_stat, ks_pval = ks_2s

#### W&B Runの終了

```python
run.finish()
```

単一のモデルをトレーニングしたので、次はハイパーパラメーター探索を使って性能を最適化してみましょう。

# ハイパーパラメーター探索

Weights and Biasesは、独自の[Sweeps機能](https://docs.wandb.ai/guides/sweeps/python-api)や[Ray Tuneインテグレーション](https://docs.wandb.ai/guides/sweeps/advanced-sweeps/ray-tune)を使ってハイパーパラメーター探索を行うことも可能です。より高度なハイパーパラメーター探索のオプションについては、[当社のドキュメント](https://docs.wandb.ai/guides/sweeps/python-api)をご覧ください。

**[こちらをクリック](https://wandb.ai/morgan/credit_score_sweeps/sweeps/iuppbs45)** して、このノートブックを使って生成された1000回のrunのsweep結果を確認してください。

#### Sweep設定の定義
まず、探索するハイパーパラメーターと使用する探索の種類を定義します。learning_rate、gamma、min_child_weights、early_stopping_roundsに対するランダム検索を行います。

```python
sweep_config = {
    "method": "random",
    "parameters": {
        "learning_rate": {"min": 0.001, "max": 1.0},
        "gamma": {"min": 0.001, "max": 1.0},
        "min_child_weight": {"min": 1, "max": 150},
        "early_stopping_rounds": {"values": [10, 20, 30, 40]},
    },
}

sweep_id = wandb.sweep(sweep_config, project=WANDB_PROJECT)
```

#### トレーニング関数の定義

次に、これらのハイパーパラメーターを使用してモデルをトレーニングする関数を定義します。runを初期化する際に`job_type='sweep'`として、必要に応じてこれらのrunをメインワークスペースから簡単にフィルタリングできるようにします。

```python
def train():
    with wandb.init(job_type="sweep") as run:
        bst_params = {
            "objective": "binary:logistic",
            "base_score": base_rate,
            "gamma": run.config["gamma"],
            "learning_rate": run.config["learning_rate"],
            "max_depth": 3,
            "min_child_weight": run.config["min_child_weight"],
            "n_estimators": 25,
            "nthread": 24,
            "random_state": 42,
            "reg_alpha": 0,
            "reg_lambda": 0,  ## 初期値: 1
            "eval_metric": ["auc", "logloss"],
            "tree_method": "hist",
        }

        # WandbCallbackを使用してXGBoostClassifierを初期化
        xgbmodel = xgb.XGBClassifier(
            **bst_params,
            callbacks=[WandbCallback()],
            early_stopping_rounds=run.config["early_stopping_rounds"]
        )

        # モデルをトレーニング
        xgbmodel.fit(trndat[p_vars], y_trn, eval_set=[(valdat[p_vars], y_val)])

        bstr = xgbmodel.get_booster()

        # boosterのメトリクスをログ
        run.summary["best_ntree_limit"] = bstr.best_ntree_limit

        # トレーニングとバリデーションの予測を取得
        trnYpreds = xgbmodel.predict_proba(trndat[p_vars])[:, 1]
        valYpreds = xgbmodel.predict_proba(valdat[p_vars])[:, 1]

        # 追加のトレーニングメトリクスをログ
        false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(
            y_trn, trnYpreds
        )
        run.summary["train_ks_stat"] = max(true_positive_rate - false_positive_rate)
        run.summary["train_auc"] = metrics.auc(false_positive_rate, true_positive_rate)
        run.summary["train_log_loss"] = -(
            y_trn * np.log(trnYpreds) + (1 - y_trn) * np.log(1 - trnYpreds)
        ).sum() / len(y_trn)

        # 追加のバリデーションメトリクスをログ
        ks_stat, ks_pval = ks_2samp(valYpreds[y_val == 1], valYpreds[y_val == 0])
        run.summary["val_ks_2samp"] = ks_stat
        run.summary["val_ks_pval"] = ks_pval
        run.summary["val_auc"] = metrics.roc_auc_score(y_val, valYpreds)
        run.summary["val_acc_0.5"] = metrics.accuracy_score(
            y_val, np.where(valYpreds >= 0.5, 1, 0)
        )
        run.summary["val_log_loss"] = -(
            y_val * np.log(valYpreds) + (1 - y_val) * np.log(1 - valYpreds)
        ).sum() / len(y_val)
```

#### Sweepsエージェントの実行

```python
count = 10  # 実行するrunの数
wandb.agent(sweep_id, function=train, count=count)
```

## お気に入りのMLライブラリですでにW&Bが使用可能に

Weights and Biasesは、以下のようなすべてのお気に入りの機械学習およびディープラーニングライブラリと連携しています：

- Pytorch Lightning
- Keras
- Hugging Face
- JAX
- Fastai
- XGBoost
- Sci-Kit Learn
- LightGBM 

**詳細は[W&B integrations](https://docs.wandb.ai/guides/integrations)をご覧ください**