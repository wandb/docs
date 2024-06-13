
# XGBoost

[**Colabノートブックで試す →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Credit_Scorecards_with_XGBoost_and_W&B.ipynb)

このノートブックでは、XGBoostモデルをトレーニングして、提出されたローン申請がデフォルトになるかどうかを分類します。XGBoostのようなブースティングアルゴリズムを使用することで、ローン評価のパフォーマンスが向上し、内部リスク管理機能や外部の規制当局にとっても解釈可能なままです。

このノートブックは、ScotiaBankのPaul EdwardsによるNvidia GTC21での講演に基づいています。[こちらのリンク](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31327/)から、XGBoostがどのようにして解釈可能なクレジットスコアカードを構築するために使用されるかをご覧いただけます。また、Stephen Denton（stephen.denton@scotiabank.com）から提供された[サンプルコード](https://github.com/rapidsai-community/showcase/tree/main/event_notebooks/GTC_2021/credit_scorecard)を使用します。

### [こちらをクリック](https://wandb.ai/morgan/credit_scorecard)して、このノートブックで作成されたライブのW&Bダッシュボードを閲覧し、操作する

# このノートブックについて

このColabでは、Weights & Biasesが規制されたEntitiesにどのように役立つかをカバーします：
- **データETL開発フロー（ローカルまたはS3やGCSのクラウドサービス）をトラッキングしてバージョン管理**
- **実験結果をトラッキングしてトレーニングされたモデルを保存**
- **複数の評価メトリクスを視覚的に検査**
- **ハイパーパラメータースイープを使用してパフォーマンスを最適化**

**実験と結果のトラッキング**

トレーニングのハイパーパラメーターと出力メトリクスをすべてトラッキングして、実験ダッシュボードを生成します：

![credit_scorecard](/images/tutorials/credit_scorecard/credit_scorecard.png)

**ハイパーパラメーター探索を実行して最適なハイパーパラメーターを見つける**

Weights & Biasesはハイパーパラメーター探索も可能にします。独自の[Sweeps機能](https://docs.wandb.ai/guides/sweeps)や[Ray Tune統合](https://docs.wandb.ai/guides/sweeps/advanced-sweeps/ray-tune)を使用できます。より高度なハイパーパラメーター探索オプションの使用方法については、ドキュメントをご覧ください。

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

## AWS S3、Google Cloud StorageおよびW&B Artifacts

![credit_scorecard_3](/images/tutorials/credit_scorecard/credit_scorecard_3.png)

Weights & Biases **Artifacts**を使用すると、エンドツーエンドのトレーニング開発フローをログに記録し、すべての実験が再現可能であることを保証します。

データのプライバシーはWeights & Biasesにとって非常に重要です。そのため、AWS S3やGoogle Cloud Storageなど、独自のプライベートクラウドからArtifactsを作成することをサポートしています。ローカルやオンプレミスのW&Bもリクエストに応じて利用可能です。

デフォルトでは、W&BはアメリカにあるプライベートGoogle Cloud StorageバケットにArtifactsファイルを保存します。すべてのファイルは保存中および転送中に暗号化されます。機密ファイルについては、プライベートW&Bインストールまたは参照アーティファクトの使用をお勧めします。

## Artifacts参照例
**S3/GCSのメタデータを含むartifactを作成**

このartifactは、S3/GCSオブジェクトのETag、サイズ、バージョンID（バケットでオブジェクトバージョン管理が有効な場合）などのメタデータのみで構成されます。

```python
run = wandb.init()
artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("s3://my-bucket/datasets/mnist")
run.log_artifact(artifact)
```

**必要に応じてArtifactをローカルにダウンロード**

W&Bは、アーティファクトがログに記録されたときのメタデータを使用して、基盤となるバケットからファイルを取得します。

```python
artifact = run.use_artifact("mnist:latest", type="dataset")
artifact_dir = artifact.download()
```

Artifacts参照の詳細、資格情報の設定については[Artifactsの参照](https://docs.wandb.ai/guides/artifacts/references)を参照してください。

## W&Bにログイン
Weights & Biasesにログイン

```python
import wandb

wandb.login()

WANDB_PROJECT = "vehicle_loan_default"
```

## 車両ローンデータセット

L&Tからの[車両ローンデフォルト予測データセット](https://www.kaggle.com/sneharshinde/ltfs-av-data)の簡略版を使用します。このデータセットはW&B Artifactsに保存されています。

```python
# データを保存するためのフォルダを指定します。フォルダが存在しない場合は新しく作成されます
data_dir = Path(".")
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

id_vars = ["UniqueID"]
targ_var = "loan_default"
```

関数をpickleする関数を作成

```python
def function_to_string(fn):
    return getsource(detect.code(fn))
```

#### W&B Artifactsからデータをダウンロード

W&B Artifactsからデータセットをダウンロードします。最初にW&Bのrunオブジェクトを作成し、それを使用してデータをダウンロードします。データがダウンロードされたら、ワンホットエンコーディングされます。この処理されたデータは、新しいArtifactとして同じW&Bにログされます。データをダウンロードしたW&Bにログすることで、この新しいArtifactを生データセットのArtifactに関連付けます。

```python
run = wandb.init(project=WANDB_PROJECT, job_type="preprocess-data")
```

W&Bから車両ローンデフォルトデータのサブセットをダウンロードします。これには、`train.csv`と`val.csv`ファイルおよびいくつかのユーティルファイルが含まれています。

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

#### データのワンホットエンコーディング

```python
# データをデータフレームにロード
dataset = pd.read_csv(data_dir / "vehicle_loans_subset.csv")

# データのワンホットエンコーディング
dataset, p_vars = one_hot_encode_data(dataset, id_vars, targ_var)

# 前処理されたデータを保存
processed_data_path = data_dir / "proc_ds.csv"
dataset.to_csv(processed_data_path, index=False)
```

#### 処理されたデータをW&B Artifactsにログ

```python
# 処理されたデータを含む新しいartifactをArtifactsに作成
processed_ds_art = wandb.Artifact(
    name="vehicle_defaults_processed",
    type="processed_dataset",
    description="ワンホットエンコーディングされたデータセット",
    metadata={"preprocessing_fn": function_to_string(one_hot_encode_data)},
)

# 処理されたデータをArtifactに添付
processed_ds_art.add_file(processed_data_path)

# このArtifactを現在のwandb runにログ
run.log_artifact(processed_ds_art)

run.finish()
```

## トレーニング/バリデーション分割

ここでは、wandb runオブジェクトを作成する別のパターンを示します。以下のセルでは、データセットを分割するコードは`wandb.init() as run`の呼び出しでラップされています。

ここで行うことは：
- wandb runを開始
- Artifactsからワンホットエンコーディングされたデータセットをダウンロード
- トレーニング/バリデーションの分割を行い、分割に使用されたパラメーターをログ
- 新しい`trndat`と`valdat`データセットをArtifactsにログ
- 自動的にwandb runを終了

```python
with wandb.init(
    project=WANDB_PROJECT, job_type="train-val-split"
) as run:  # configはここでは任意です
    # W&Bから車両ローンデフォルトデータのサブセットをダウンロード
    dataset_art = run.use_artifact(
        "vehicle_defaults_processed:latest", type="processed_dataset"
    )
    dataset_dir = dataset_art.download(data_dir)
    dataset = pd.read_csv(processed_data_path)

    # 分割パラメーターを設定
    test_size = 0.25
    random_state = 42

    # 分割パラメーターをログ
    run.config.update({"test_size": test_size, "random_state": random_state})

    # トレーニング/バリデーションの分割を行う
    trndat, valdat = model_selection.train_test_split(
        dataset,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset[[targ_var]],
    )

    print(f"トレーニングデータセットのサイズ: {trndat[targ_var].value_counts()} \n")
    print(f"バリデーションデータセットのサイズ: {valdat[targ_var].value_counts()}")

    # 分割されたデータセットを保存
    train_path = data_dir / "train.csv"
    val_path = data_dir / "val.csv"
    trndat.to_csv(train_path, index=False)
    valdat.to_csv(val_path, index=False)

    # 処理されたデータを含む新しいartifactをArtifactsに作成
    split_ds_art = wandb.Artifact(
        name="vehicle_defaults_split",
        type="train-val-dataset",
        description="トレーニングおよびバリデーションに分割された処理済みデータセット",
        metadata={"test_size": test_size, "random_state": random_state},
    )

    # 処理されたデータをArtifactに添付
    split_ds_art.add_file(train_path)
    split_ds_art.add_file(val_path)

    # Artifactをログ
    run.log_artifact(split_ds_art)
```

#### トレーニングデータセットの検査
トレーニングデータセットの概要を確認

```python
trndict = describe_data_g_targ(trndat, targ_var)
trndat.head()
```

### W&B Tablesでデータセットをログ

W&B Tablesを使用すると、画像、ビデオ、オーディオなどのリッチメディアを含むタブularデータをログ、クエリ、分析できます。これにより、データセットを理解し、モデルの予測を視覚化し、洞察を共有できます。詳細は[W&B Tablesガイド](https://docs.wandb.ai/guides/tables)をご覧ください。

```python
# wandb runを作成し、オプションで"log-dataset"ジョブタイプを指定して整理を保つ
run = wandb.init(
    project=WANDB_PROJECT, job_type="log-dataset"
)  # configは任意です

# W&Bテーブルを作成し、ランダムに選択したデータセットの1000行をログ
table = wandb.Table(dataframe=trndat.sample(1000))

# W&Bワークスペースにテーブルをログ
wandb.log({"processed_dataset": table})

# wandb runを終了
wandb.finish()
```

# モデリング

## XGBoostモデルのフィット

今度は、車両ローン申請がデフォルトになるかどうかを分類するためにXGBoostモデルをフィットします。

### GPUでのトレーニング
XGBoostモデルをGPUでトレーニングしたい場合は、次のパラメータをXGBoostに渡してください：

```python
"tree_method": "gpu_hist"
```

#### 1) W&B Runを初期化

```python
run = wandb.init(project=WANDB_PROJECT, job_type="train-model")
```

#### 2) モデルパラメーターの設定とログ

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
    "tree_method": "hist",  # GPUでトレーニングするには `gpu_hist`
}
```

XGBoostのトレーニングパラメーターをW&B run configにログ

```python
run.config.update(dict(bst_params))
run.config.update({"early_stopping_rounds": early_stopping_rounds})
```

#### 3) W&B Artifactsからトレーニングデータをロード

```python
# Artifactsからトレーニングデータをロード
trndat, valdat = load_training_data(
    run=run, data_dir=data_dir, artifact_name="vehicle_defaults_split:latest"
)

## ターゲット列をシリーズとして抽出
y_trn = trndat.loc[:, targ_var].astype(int)
y_val = valdat.loc[:, targ_var].astype(int)
```

#### 4) モデルのフィット、結果をW&Bにログ、およびモデルをW&B Artifactsに保存

すべてのXGBoostモデルパラメーターをログするために`WandbCallback`を使用します。これは、詳細については[W&Bドキュメント](https://docs.wandb.ai/guides/integrations)をご覧ください。他のライブラリとしての統合もあり、LightGBMなども含まれています。

```python
from wandb.integration.xgboost import WandbCallback

# WandbCallbackを使用してXGBoostClassifierを初期化
xgbmodel = xgb.XGBClassifier(
    **bst_params,
    callbacks=[WandbCallback(log_model=True)],
    early_stopping_rounds=run.config["early_stopping_rounds"]
)

# モデルのトレーニング
xgbmodel.fit(trndat[p_vars], y_trn, eval_set=[(valdat[p_vars], y_val)])
```

#### 5) 追加のトレーニングおよび評価メトリクスをW&Bにログ

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
ks_stat, ks_pval = ks_2samp(valYpreds[y_val == 1], valYpreds[y_val == 0])
run.summary["val_ks_2samp"] = ks_stat
run.summary["val_ks_pval"] = ks_pval
run.summary["val_auc