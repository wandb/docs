


# XGBoost

[**Try in a Colab Notebook here →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Credit_Scorecards_with_XGBoost_and_W&B.ipynb)

このノートブックでは、XGBoostモデルをトレーニングして、提出されたローン申請がデフォルトするかどうかを分類します。XGBoostのようなブースティングアルゴリズムを使用することで、ローン評価のパフォーマンスが向上し、内部のリスク管理機能や外部の規制当局に対しても解釈可能性を維持できます。

このノートブックは、ScotiaBankのPaul Edwards氏によるNvidia GTC21の講演に基づいており、XGBoostが解釈可能でパフォーマンスの高いクレジットスコアカードを構築する方法を紹介しました。彼らはまた[サンプルコードを共有](https://github.com/rapidsai-community/showcase/tree/main/event_notebooks/GTC_2021/credit_scorecard)してくださり、このコードを通して使用します。ScotiabankのStephen Denton (stephen.denton@scotiabank.com)氏に感謝します。

### [こちらをクリック](https://wandb.ai/morgan/credit_scorecard)して、このノートブックで構築されたライブのW&B ダッシュボードを表示および操作します

# このノートブックで

このColabノートで、Weights and Biasesを使用して規制対象エンティティが以下のことをどのように行うかを紹介します。
- **データETLパイプラインのトラッキングとバージョン管理**（ローカルまたはS3やGCSのようなクラウドサービスで）
- **実験結果のトラッキング**およびトレーニング済みモデルの保存
- **複数の評価メトリクスの視覚的検査**
- **ハイパーパラメータースイープによるパフォーマンスの最適化**

**実験と結果のトラッキング**

すべてのトレーニングハイパーパラメーターと出力メトリクスをトラッキングし、Experiments ダッシュボードを生成します。

![credit_scorecard](/images/tutorials/credit_scorecard/credit_scorecard.png)

**ベストハイパーパラメーターを見つけるためのハイパーパラメータースイープの実行**

Weights and Biasesは、独自の[Sweeps機能](https://docs.wandb.ai/guides/sweeps)や[Ray Tune統合](https://docs.wandb.ai/guides/sweeps/advanced-sweeps/ray-tune)を使用して、ハイパーパラメータースイープを実行することもできます。より高度なハイパーパラメータースイープオプションの使用方法については、ドキュメントを参照してください。

![credit_scorecard_2](/images/tutorials/credit_scorecard/credit_scorecard_2.png)

# 設定

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

## AWS S3、Google Cloud Storage、W&B Artifacts

![credit_scorecard_3](/images/tutorials/credit_scorecard/credit_scorecard_3.png)

Weights and Biasesの**Artifacts**は、エンドツーエンドのトレーニングパイプラインをログに記録し、実験が常に再現可能であることを保証します。

データプライバシーはWeights & Biasesにとって重要であり、AWS S3やGoogle Cloud StorageなどのプライベートクラウドからArtifactsを作成することをサポートしています。ローカルおよびオンプレミスのW&Bもリクエストに応じて利用可能です。

デフォルトでは、W&Bはアメリカに位置するプライベートGoogle Cloud Storageバケットにアーティファクトファイルを保存します。すべてのファイルは保存時および転送時に暗号化されます。機密性の高いファイルには、プライベートW&Bインストールまたは参照アーティファクトの使用を推奨します。

## アーティファクト参照例
**S3/GCSメタデータを持つアーティファクトを作成**

アーティファクトは、S3/GCSオブジェクトのETag、サイズ、バージョンID（バケットにオブジェクトバージョン管理が有効な場合）などのメタデータで構成されています。

```python
run = wandb.init()
artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("s3://my-bucket/datasets/mnist")
run.log_artifact(artifact)
```

**必要なときにアーティファクトをローカルにダウンロード**

W&Bは、アーティファクトログ時に記録されたメタデータを使用して、基礎となるバケットからファイルを取得します。

```python
artifact = run.use_artifact("mnist:latest", type="dataset")
artifact_dir = artifact.download()
```

参照を使ったArtifactsの使用方法、資格情報の設定などについては、[Artifact References](https://docs.wandb.ai/guides/artifacts/references)を参照してください。

## W&Bにログイン
Weights and Biasesにログインします

```python
import wandb

wandb.login()

WANDB_PROJECT = "vehicle_loan_default"
```

## 車両ローンデータセット

L&Tの[Vehicle Loan Default Prediction dataset](https://www.kaggle.com/sneharshinde/ltfs-av-data)の簡略化されたバージョンを使用します。このデータセットはW&B Artifactsに保存されています。

```python
# データを保存するフォルダを指定します。存在しない場合は新しいフォルダが作成されます。
data_dir = Path(".")
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

id_vars = ["UniqueID"]
targ_var = "loan_default"
```

関数をピクル化する関数を作成します

```python
def function_to_string(fn):
    return getsource(detect.code(fn))
```

#### W&B Artifactsからデータをダウンロード

W&B Artifactsからデータセットをダウンロードします。最初にW&Bのrunオブジェクトを作成し、これを使用してデータをダウンロードします。データがダウンロードされると、ワンホットエンコードされます。この処理済みデータは、新しいArtifactとして同じW&Bにログされます。

```python
run = wandb.init(project=WANDB_PROJECT, job_type="preprocess-data")
```

W&Bから車両ローンデフォルトデータのサブセットをダウンロードします。これには`train.csv`および`val.csv`ファイルや一部のユーティリティファイルが含まれています。

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
# データをDataframeに読み込み
dataset = pd.read_csv(data_dir / "vehicle_loans_subset.csv")

# データのワンホットエンコード
dataset, p_vars = one_hot_encode_data(dataset, id_vars, targ_var)

# 前処理されたデータを保存
processed_data_path = data_dir / "proc_ds.csv"
dataset.to_csv(processed_data_path, index=False)
```

#### 処理済みデータをW&B Artifactsにログ

```python
# 処理済みデータの新しいアーティファクトを作成し、関数を含めてArtifactsに追加
processed_ds_art = wandb.Artifact(
    name="vehicle_defaults_processed",
    type="processed_dataset",
    description="One-hot encoded dataset",
    metadata={"preprocessing_fn": function_to_string(one_hot_encode_data)},
)

# 処理済みデータをArtifactに追加
processed_ds_art.add_file(processed_data_path)

# このArtifactを現在のwandb runにログ
run.log_artifact(processed_ds_art)

run.finish()
```

## トレーニング/バリデーション分割の取得

ここでは、wandb runオブジェクトを作成するための別のパターンを示します。以下のセルでは、データセットを分割するコードが`wandb.init() as run`の呼び出しでラップされています。

ここでは以下を行います。
- wandb runを開始
- Artifactsからワンホットエンコードされたデータセットをダウンロード
- 分割に使用したパラメータをログ
- 新しい`trndat`および`valdat`データセットをArtifactsにログ
- wandb runを自動的に終了

```python
with wandb.init(
    project=WANDB_PROJECT, job_type="train-val-split"
) as run:  # configはオプションです

  # W&Bから車両ローンデフォルトデータのサブセットをダウンロード
  dataset_art = run.use_artifact(
      "vehicle_defaults_processed:latest", type="processed_dataset"
  )
  dataset_dir = dataset_art.download(data_dir)
  dataset = pd.read_csv(processed_data_path)

  # 分割パラメータを設定
  test_size = 0.25
  random_state = 42

  # 分割パラメータをログ
  run.config.update({"test_size": test_size, "random_state": random_state})

  # トレーニング/バリデーション分割を実行
  trndat, valdat = model_selection.train_test_split(
      dataset,
      test_size=test_size,
      random_state=random_state,
      stratify=dataset[[targ_var]],
  )

  print(f"Train dataset size: {trndat[targ_var].value_counts()} \n")
  print(f"Validation dataset sizeL {valdat[targ_var].value_counts()}")

  # 分割されたデータセットを保存
  train_path = data_dir / "train.csv"
  val_path = data_dir / "val.csv"
  trndat.to_csv(train_path, index=False)
  valdat.to_csv(val_path, index=False)

  # 処理済みデータの新しいアーティファクトを作成し、関数を含めてArtifactsに追加
  split_ds_art = wandb.Artifact(
      name="vehicle_defaults_split",
      type="train-val-dataset",
      description="Processed dataset split into train and valiation",
      metadata={"test_size": test_size, "random_state": random_state},
  )

  # 処理済みデータをArtifactに追加
  split_ds_art.add_file(train_path)
  split_ds_art.add_file(val_path)

  # Artifactをログ
  run.log_artifact(split_ds_art)
```

#### トレーニングデータセットの検査
トレーニングデータセットの概要を取得

```python
trndict = describe_data_g_targ(trndat, targ_var)
trndat.head()
```

### W&B Tablesでデータセットをログ

W&B Tablesを使用すると、画像、ビデオ、音声などのリッチメディアを含む表形式のデータをログ、クエリ、および分析できます。これを使用してデータセットを理解し、モデルの予測を可視化し、洞察を共有することができます。詳細については、[W&B Tables Guide](https://docs.wandb.ai/guides/tables)をご覧ください。

```python
# "log-dataset"ジョブタイプを指定して、W&B runを作成（オプション）
run = wandb.init(
    project=WANDB_PROJECT, job_type="log-dataset"
)  # configはオプションです

# データセットのランダムに選んだ1000行をW&B Tableにログ
table = wandb.Table(dataframe=trndat.sample(1000))

# W&B workspaceにTableをログ
wandb.log({"processed_dataset": table})

# wandb runを閉じる
wandb.finish()
```

# モデリング

## XGBoostモデルのフィット

ここでは、XGBoostモデルをトレーニングして、車両ローン申請がデフォルトになるかどうかを分類します。

### GPU上でのトレーニング
XGBoostモデルをGPU上でトレーニングしたい場合は、XGBoostに渡すパラメータで以下を設定するだけです。

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
    "gamma": 1,  # デフォルト: 0
    "learning_rate": 0.1,  # デフォルト: 0.1
    "max_depth": 3,
    "min_child_weight": 100,  # デフォルト: 1
    "n_estimators": 25,
    "nthread": 24,
    "random_state": 42,
    "reg_alpha": 0,
    "reg_lambda": 0,  # デフォルト: 1
    "eval_metric": ["auc", "logloss"],
    "tree_method": "hist",  # GPUでトレーニングするには`gpu_hist`を使用
}
```

xgboostのトレーニングパラメータをW&Bのrun configにログ

```python
run.config.update(dict(bst_params))
run.config.update({"early_stopping_rounds": early_stopping_rounds})
```

#### 3) W&B Artifactsからトレーニングデータをロード

```python
# トレーニングデータをArtifactsからロード
trndat, valdat = load_training_data(
    run=run, data_dir=data_dir, artifact_name="vehicle_defaults_split:latest"
)

# 目標列をSeriesとして抽出
y_trn = trndat.loc[:, targ_var].astype(int)
y_val = valdat.loc[:, targ_var].astype(int)
```

#### 4) モデルのフィット、結果のログ、モデルをW&B Artifactsに保存

すべてのxgboostモデルパラメータをログするために、`WandbCallback`を使用しました。詳細は[W&Bのドキュメント](https://docs.wandb.ai/guides/integrations)を参照してください。他のライブラリも対応しています。

```python
from wandb.integration.xgboost import WandbCallback

# WandbCallback付きのXGBoostClassifierを初期化
xgbmodel = xgb.XGBClassifier(
    **bst_params,
    callbacks=[WandbCallback(log_model=True)],
    early_stopping_rounds=run.config["early_stopping_rounds"]
)

# モデルのトレーニング
xgbmodel.fit(trndat[p_vars], y_trn, eval_set=[(valdat[p_vars], y_val)])
```

#### 5) 追加のトレーニングと評価メトリクスをW&Bにログ

```python
bstr = xgbmodel.get_booster()

# トレーニングおよびバリデーション予測を取得
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
    y_val * np.log(valYpreds) + (1 -