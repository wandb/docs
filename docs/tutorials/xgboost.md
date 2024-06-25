
# XGBoost

[**Try in a Colab Notebook here →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Credit_Scorecards_with_XGBoost_and_W&B.ipynb)

このノートブックでは、XGBoostモデルをトレーニングして、提出されたローン申請がデフォルトするかどうかを分類します。XGBoostのようなブースティングアルゴリズムを使用することで、ローン評価のパフォーマンスを向上させ、内部のリスク管理機能や外部の規制機関に対する解釈性を維持します。

このノートブックは、ScotiaBankのPaul EdwardsがNvidia GTC21で行った講演に基づいています。彼はXGBoostを使用して、より高性能で解釈可能なクレジットスコアカードを構築する方法を[発表](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31327/)しました。彼らはまた、ScotiabankのStephen Denton（stephen.denton@scotiabank.com）が公開してくれた[サンプルコード](https://github.com/rapidsai-community/showcase/tree/main/event_notebooks/GTC_2021/credit_scorecard)も共有してくれました。

### [こちらをクリック](https://wandb.ai/morgan/credit_scorecard)して、このノートブックで構築されたライブW&Bダッシュボードを閲覧し、対話する

# このノートブックで

このcolabでは、Weights and Biasesが規制対象のエンティティをどのように支援するかを紹介します。
- **データETLパイプラインのトラッキングとバージョン管理**（ローカルまたはS3やGCSのクラウドサービスで）
- **実験結果のトラッキング**とトレーニング済みモデルの保存
- **複数の評価メトリクスを視覚的に検査**
- **ハイパーパラメータ探索でパフォーマンスを最適化**

**Experiments と Results のトラッキング**

すべてのトレーニングハイパーパラメーターと出力メトリクスをトラッキングし、Experiments Dashboardを生成します:

![credit_scorecard](/images/tutorials/credit_scorecard/credit_scorecard.png)

**ハイパーパラメータ探索を実行して最適なハイパーパラメータを見つける**

Weights and Biasesは、独自の[Sweeps機能](https://docs.wandb.ai/guides/sweeps)や[Ray Tune integration](https://docs.wandb.ai/guides/sweeps/advanced-sweeps/ray-tune)を使用して、ハイパーパラメータ探索を実行できます。より高度なハイパーパラメータ探索オプションの使用方法については、ドキュメントをご覧ください。

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

## AWS S3、Google Cloud Storage、およびW&B Artifacts

![credit_scorecard_3](/images/tutorials/credit_scorecard/credit_scorecard_3.png)

Weights and Biasesの**Artifacts**を使用すると、エンドツーエンドのトレーニングパイプラインをログに記録し、実験が常に再現可能であることを保証できます。

Weights & Biasesにとってデータプライバシーは重要であるため、AWS S3やGoogle Cloud Storageなどのプライベートクラウドからのリファレンスを使用してArtifactsを作成することをサポートしています。ローカル、オンプレミスのW&Bもリクエストに応じて利用可能です。

デフォルトでは、W&Bはアメリカ合衆国に位置するプライベートなGoogle Cloud Storageバケットにアーティファクトファイルを保存します。すべてのファイルは保存中と転送中に暗号化されます。機密性の高いファイルについては、プライベートW&Bインストールやリファレンスアーティファクトの使用をお勧めします。

## Artifactsリファレンスの例
**S3/GCSメタデータを使用してアーティファクトを作成する**

アーティファクトは、S3/GCSオブジェクトに関するメタデータ（ETag、サイズ、バージョンID（バケットにオブジェクトバージョニングが有効な場合））のみで構成されます。

```python
run = wandb.init()
artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("s3://my-bucket/datasets/mnist")
run.log_artifact(artifact)
```

**必要に応じてアーティファクトをローカルにダウンロードする**

W&Bは、アーティファクトがログに記録されたときに記録されたメタデータを使用して、ファイルを基盤となるバケットから取得します。

```python
artifact = run.use_artifact("mnist:latest", type="dataset")
artifact_dir = artifact.download()
```

Artifactsのリファレンス、認証情報の設定などの詳細については、[Artifact References](https://docs.wandb.ai/guides/artifacts/references)をご覧ください。

## W&Bにログインする
Weights and Biasesにログイン


```python
import wandb

wandb.login()

WANDB_PROJECT = "vehicle_loan_default"
```

## Vehicle Loan Dataset

L&Tの[Vehicle Loan Default Prediction dataset](https://www.kaggle.com/sneharshinde/ltfs-av-data)の簡略版を使用します。このデータセットはW&B Artifactsに保存されています。


```python
# データを保存するフォルダを指定、新しいフォルダが存在しない場合は作成されます
data_dir = Path(".")
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

id_vars = ["UniqueID"]
targ_var = "loan_default"
```

関数をピクル化する関数を作成する


```python
def function_to_string(fn):
    return getsource(detect.code(fn))
```

#### W&B Artifactsからデータをダウンロードする

私たちはW&B Artifactsからデータセットをダウンロードします。まず、データをダウンロードするために使用するW&B run オブジェクトを作成する必要があります。データがダウンロードされると、それはワンホットエンコードされます。この処理されたデータは、新しいArtifactとして同じW&Bにログに記録されます。データをダウンロードしたW&Bにログを記録することで、この新しいArtifactを生データセットのArtifactに結びつけます。


```python
run = wandb.init(project=WANDB_PROJECT, job_type="preprocess-data")
```

W&Bから車両ローンデフォルトデータのサブセットをダウンロードします。これには、`train.csv`と`val.csv`ファイル、およびいくつかのユーティリティファイルが含まれます。


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

#### データのワンホットエンコード


```python
# データをDataFrameにロードする
dataset = pd.read_csv(data_dir / "vehicle_loans_subset.csv")

# データをワンホットエンコードする
dataset, p_vars = one_hot_encode_data(dataset, id_vars, targ_var)

# 前処理されたデータを保存する
processed_data_path = data_dir / "proc_ds.csv"
dataset.to_csv(processed_data_path, index=False)
```

#### 処理されたデータをW&B Artifactsにログに記録する


```python
# 処理されたデータと、それを作成した関数を含む、新しいアーティファクトをArtifactsに作成する
processed_ds_art = wandb.Artifact(
    name="vehicle_defaults_processed",
    type="processed_dataset",
    description="ワンホットエンコードされたデータセット",
    metadata={"preprocessing_fn": function_to_string(one_hot_encode_data)},
)

# 前処理されたデータをアーティファクトに添付する
processed_ds_art.add_file(processed_data_path)

# このArtifactを現在のwandb runにログする
run.log_artifact(processed_ds_art)

run.finish()
```

## トレーニング/バリデーション分割を取得する

ここでは、wandb runオブジェクトを作成するための別のパターンを示します。以下のセルでは、データセットを分割するコードを`wandb.init() as run`の呼び出しでラップしています。

ここで行うことは：
- wandb runを開始する
- Artifactsからワンホットエンコードされたデータセットをダウンロードする
- Train/Val分割を行い、その際のパラメーターをログに記録する
- 新しい `trndat` と `valdat` データセット をArtifactsにログする
- 自動的にwandb runを終了する


```python
with wandb.init(
    project=WANDB_PROJECT, job_type="train-val-split"
) as run:  # configはここではオプションです
    # W&Bから車両ローンデフォルトデータのサブセットをダウンロードする
    dataset_art = run.use_artifact(
        "vehicle_defaults_processed:latest", type="processed_dataset"
    )
    dataset_dir = dataset_art.download(data_dir)
    dataset = pd.read_csv(processed_data_path)

    # 分割パラメーターを設定する
    test_size = 0.25
    random_state = 42

    # 分割パラメーターをログに記録する
    run.config.update({"test_size": test_size, "random_state": random_state})

    # Train/Val分割を行う
    trndat, valdat = model_selection.train_test_split(
        dataset,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset[[targ_var]],
    )

    print(f"トレーニングデータセットサイズ: {trndat[targ_var].value_counts()} \n")
    print(f"バリデーションデータセットサイズ: {valdat[targ_var].value_counts()}")

    # 分割されたデータセットを保存する
    train_path = data_dir / "train.csv"
    val_path = data_dir / "val.csv"
    trndat.to_csv(train_path, index=False)
    valdat.to_csv(val_path, index=False)

    # 処理されたデータと、それを作成した関数を含む、新しいアーティファクトをArtifactsに作成する
    split_ds_art = wandb.Artifact(
        name="vehicle_defaults_split",
        type="train-val-dataset",
        description="トレーニングとバリデーションに分割された処理済データセット",
        metadata={"test_size": test_size, "random_state": random_state},
    )

    # 分割されたデータをアーティファクトに添付する
    split_ds_art.add_file(train_path)
    split_ds_art.add_file(val_path)

    # アーティファクトをログに記録する
    run.log_artifact(split_ds_art)
```

#### トレーニングデータセットを調査する
トレーニングデータセットの概要を取得する


```python
trndict = describe_data_g_targ(trndat, targ_var)
trndat.head()
```

### W&B Tablesでデータセットをログする

W&B Tablesを使用すると、画像、ビデオ、音声などのリッチメディアを含む表形式のデータをログ、クエリ、分析できます。これにより、データセットを理解し、モデルの予測を視覚化し、インサイトを共有できます。詳細については、[W&B Tables Guide](https://docs.wandb.ai/guides/tables)をご覧ください


```python
# データセットのサイズを指定する。ここでは、"log-dataset"というジョブタイプを指定して整理しています
run = wandb.init(
    project=WANDB_PROJECT, job_type="log-dataset"
)  # configはここではオプションです

# W&Bテーブルを作成し、データセットの1000ランダム行をログして調査する
table = wandb.Table(dataframe=trndat.sample(1000))

# W&Bワークスペースにテーブルをログする
wandb.log({"processed_dataset": table})

# wandb runを閉じる
wandb.finish()
```

# モデリング

## XGBoostモデルのフィット

続いて、XGBoostモデルをトレーニングして、車両ローンの申請がデフォルトするかどうかを分類します

### GPUでのトレーニング
XGBoostモデルをGPUでトレーニングしたい場合、XGBoostに渡すパラメータに以下を設定します。

```python
"tree_method": "gpu_hist"
```

#### 1) W&B Runの初期化


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
    "gamma": 1,  ## デフォルト: 0
    "learning_rate": 0.1,  ## デフォルト: 0.1
    "max_depth": 3,
    "min_child_weight": 100,  ## デフォルト: 1
    "n_estimators": 25,
    "nthread": 24,
    "random_state": 42,
    "reg_alpha": 0,
    "reg_lambda": 0,  ## デフォルト: 1
    "eval_metric": ["auc", "logloss"],
    "tree_method": "hist",  # GPUでトレーニングするには`gpu_hist`を使用
}
```

XGBoostトレーニングパラメータをW&B runの設定にログ


```python
run.config.update(dict(bst_params))
run.config.update({"early_stopping_rounds": early_stopping_rounds})
```

#### 3) W&B Artifactsからトレーニングデータをロードする


```python
# Artifactsからトレーニングデータをロードする
trndat, valdat = load_training_data(
    run=run, data_dir=data_dir, artifact_name="vehicle_defaults_split:latest"
)

## 目的変数列をシリーズとして抽出する
y_trn = trndat.loc[:, targ_var].astype(int)
y_val = valdat.loc[:, targ_var].astype(int)
```

#### 4) モデルをフィットさせ、結果をW&Bにログし、モデルをW&B Artifactsに保存する

すべてのXGBoostモデルパラメータをログするために`WandbCallback`を使用します。これについては、他のライブラリの統合についても記載している[W&Bドキュメント](https://docs.wandb.ai/guides/integrations)を参照してください。


```python
from wandb.integration.xgboost import WandbCallback

# WandbCallbackでXGBoostClassifierを初期化する
xgbmodel = xgb.XGBClassifier(
    **bst_params,
    callbacks=[WandbCallback(log_model=True)],
    early_stopping_rounds=run.config["early_stopping_rounds"]
)

# モデルをトレーニングする
xgbmodel.fit(trndat[p_vars], y_trn, eval_set=[(valdat[p_vars], y_val)])
```

#### 5) 追加のトレーニングおよび評価メトリクスをW&Bにログする


```python
bstr = xgbmodel.get_booster()

# トレーニングとバリデーションの予測を取得
trnYpreds = xgbmodel.predict_proba(trndat[p_vars])[:, 1]
valYpreds = xgbmodel.predict_proba(valdat[p_vars])[:, 1]

# 追加のトレーニングメトリクスをログする
false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(
    y_trn, trnYpreds
)
run.summary["train_ks_stat"] = max(true_positive_rate - false_positive_rate)
run.summary["train_auc"] = metrics.auc(false_positive_rate, true_positive_rate)
run.summary["train_log_loss"] = -(
    y_trn * np.log(trnYpreds) + (1 - y_trn) * np.log(1 - trnYpreds)
).sum() / len(y_trn)

# 追加のバリデーションメトリクスを