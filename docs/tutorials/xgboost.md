# XGBoost

[**Colabノートブックで試す →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Credit_Scorecards_with_XGBoost_and_W&B.ipynb)

このノートブックでは、XGBoostモデルをトレーニングして、提出されたローン申請がデフォルトするかどうかを分類します。XGBoostのようなブースティングアルゴリズムを使用すると、より高いパフォーマンスのローン評価が可能になりますが、内部のリスク管理機能や外部の規制当局に対する説明可能性も保持されます。

このノートブックは、Nvidia GTC21でScotiaBankのPaul Edwardsが行った講演に基づいており、XGBoostを使用してよりパフォーマンスの高いクレジットスコアカードを構築しつつも解釈可能な状態を維持する方法が[紹介](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31327/)されています。彼らはまた、[サンプルコード](https://github.com/rapidsai-community/showcase/tree/main/event_notebooks/GTC_2021/credit_scorecard)も共有してくださり、このノートブック全体で使用します。このコードを公開してくれたScotiabankのStephen Denton（stephen.denton@scotiabank.com）に感謝します。

### [ここをクリック](https://wandb.ai/morgan/credit_scorecard)して、このノートブックで作成されたライブW&Bダッシュボードを表示および操作

# ノートブックの内容

このColabでは、Weights and Biasesが規制を受けたエンティティにどのように対応するかをカバーします:
- **データETLパイプラインの追跡とバージョン管理**（ローカルまたはS3やGCSなどのクラウドサービスで）
- **実験結果の追跡**とトレーニング済みモデルの保存
- **複数の評価指標を視覚的に検査**
- **ハイパーパラメーター探索**によるパフォーマンスの最適化

**実験と結果の追跡**

すべてのトレーニングハイパーパラメーターと出力メトリクスを追跡して、Experimentsダッシュボードを生成します:

![credit_scorecard](/images/tutorials/credit_scorecard/credit_scorecard.png)

**ハイパーパラメータースイープを実行して最適なハイパーパラメーターを見つける**

Weights and Biasesでは、独自の[Sweeps機能](https://docs.wandb.ai/guides/sweeps)や[Ray Tuneとの統合](https://docs.wandb.ai/guides/sweeps/advanced-sweeps/ray-tune)を使用してハイパーパラメータースイープを実行することができます。高度なハイパーパラメータースイープオプションの使用方法についての完全なガイドはドキュメントをご覧ください。

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

Weights and Biasesの**Artifacts**を使用すると、エンドツーエンドのトレーニングパイプラインをログに記録して、実験が常に再現可能であることを確認できます。

データプライバシーはWeights & Biasesにとって非常に重要であり、自分のプライベートクラウド（AWS S3やGoogle Cloud Storageなど）を参照元として使用してArtifactsを作成することもサポートしています。ローカルやオンプレミスのW&Bもリクエストに応じて利用可能です。

デフォルトでは、W&BはアメリカにあるプライベートなGoogle Cloud Storageバケットにアーティファクトファイルを保存します。すべてのファイルは保存時と転送時に暗号化されます。機密ファイルの場合は、プライベートなW&Bインストールか参照アーティファクトの使用をお勧めします。

## アーティファクト参照例
**S3/GCSメタデータを使用してアーティファクトを作成する**

アーティファクトは、S3/GCSオブジェクトに関するメタデータ（ETag、サイズ、バージョンIDなど）のみで構成されます（バケットでオブジェクトバージョニングが有効な場合）。

```python
run = wandb.init()
artifact = wandb.Artifact("mnist", type="dataset")
artifact.add_reference("s3://my-bucket/datasets/mnist")
run.log_artifact(artifact)
```

**必要なときにアーティファクトをローカルにダウンロードする**

W&Bは、アーティファクトがログに記録されたときに記録されたメタデータを使用して、基になるバケットからファイルを取得します。

```python
artifact = run.use_artifact("mnist:latest", type="dataset")
artifact_dir = artifact.download()
```

Artifactsの参照の使用方法、資格情報の設定などについては、[Artifactsの参照](https://docs.wandb.ai/guides/artifacts/references)をご覧ください。

## W&Bにログイン
Weights and Biasesにログインする

```python
import wandb

wandb.login()

WANDB_PROJECT = "vehicle_loan_default"
```

## 車両ローンデータセット

L&Tの[Vehicle Loan Default Prediction データセット](https://www.kaggle.com/sneharshinde/ltfs-av-data)の簡略版を使用します。このデータセットはW&B Artifactsに保存されています。

```python
# データを保存するフォルダーを指定し、存在しない場合は新しいフォルダーを作成します
data_dir = Path(".")
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

id_vars = ["UniqueID"]
targ_var = "loan_default"
```

関数をピクル関数として作成する

```python
def function_to_string(fn):
    return getsource(detect.code(fn))
```

#### W&B Artifactsからデータをダウンロードする

W&B Artifactsからデータセットをダウンロードします。最初にW&B runオブジェクトを作成し、それを使用してデータをダウンロードします。データがダウンロードされたら、ワンホットエンコードされます。この処理済みデータは、新しいArtifactとして同じW&Bにログされます。データをダウンロードしたW&Bにログすることで、この新しいArtifactを生のデータセットArtifactに結びつけます。

```python
run = wandb.init(project=WANDB_PROJECT, job_type="preprocess-data")
```

W&Bから車両ローンデフォルトデータのサブセットをダウンロードします。これには`train.csv`および`val.csv`ファイルといくつかのユーティリティファイルが含まれます。

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

#### データをワンホットエンコードする

```python
# データをDataframeにロードする
dataset = pd.read_csv(data_dir / "vehicle_loans_subset.csv")

# データをワンホットエンコードする
dataset, p_vars = one_hot_encode_data(dataset, id_vars, targ_var)

# 処理済みデータを保存する
processed_data_path = data_dir / "proc_ds.csv"
dataset.to_csv(processed_data_path, index=False)
```

#### 処理済みデータをW&B Artifactsにログする

```python
# 処理済みデータのための新しいアーティファクトを作成し、作成した関数を含めてArtifactsにログする
processed_ds_art = wandb.Artifact(
    name="vehicle_defaults_processed",
    type="processed_dataset",
    description="One-hot encoded dataset",
    metadata={"preprocessing_fn": function_to_string(one_hot_encode_data)},
)

# 処理済みデータをArtifactに添付する
processed_ds_art.add_file(processed_data_path)

# このArtifactを現在のwandb runにログする
run.log_artifact(processed_ds_art)

run.finish()
```

## トレイン/バリデーション分割を取得する

ここでは、wandb runオブジェクトを作成するための別のパターンを紹介します。以下のセルでは、データセットを分割するコードが`wandb.init() as run`の呼び出しでラップされています。

ここで行うことは次の通りです:

- wandb runを開始する
- Artifactsからワンホットエンコードされたデータセットをダウンロードする
- トレイン/バリデーション分割を行い、分割に使用したパラメータをログする
- 新しい `trndat`および`valdat` データセットをArtifactsにログする
- wandbのランを自動的に終了する

```python
with wandb.init(
    project=WANDB_PROJECT, job_type="train-val-split"
) as run:  # configは任意です
    # W&Bから車両ローンデフォルトデータのサブセットをダウンロードする
    dataset_art = run.use_artifact(
        "vehicle_defaults_processed:latest", type="processed_dataset"
    )
    dataset_dir = dataset_art.download(data_dir)
    dataset = pd.read_csv(processed_data_path)

    # Split Paramsを設定する
    test_size = 0.25
    random_state = 42

    # 分割パラメータをログする
    run.config.update({"test_size": test_size, "random_state": random_state})

    # Train/Val Splitを行う
    trndat, valdat = model_selection.train_test_split(
        dataset,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset[[targ_var]],
    )

    print(f"トレインデータセットのサイズ: {trndat[targ_var].value_counts()} \n")
    print(f"バリデーションデータセットのサイズ: {valdat[targ_var].value_counts()}")

    # 分割データセットを保存する
    train_path = data_dir / "train.csv"
    val_path = data_dir / "val.csv"
    trndat.to_csv(train_path, index=False)
    valdat.to_csv(val_path, index=False)

    # 処理済みデータのための新しいアーティファクトを作成し、作成した関数を含めてArtifactsにログする
    split_ds_art = wandb.Artifact(
        name="vehicle_defaults_split",
        type="train-val-dataset",
        description="TrainとValidationに分割された処理済みデータセット",
        metadata={"test_size": test_size, "random_state": random_state},
    )

    # 処理済みデータをArtifactに添付する
    split_ds_art.add_file(train_path)
    split_ds_art.add_file(val_path)

    # Artifactをログする
    run.log_artifact(split_ds_art)
```

#### トレーニングデータセットを検査する
トレーニングデータセットの概要を取得する

```python
trndict = describe_data_g_targ(trndat, targ_var)
trndat.head()
```

### W&B Tablesを使用してデータセットをログする

W&B Tablesを使用すると、画像、ビデオ、オーディオなどのリッチメディアを含む表形式のデータをログ、クエリ、および分析できます。これを使用してデータセットを理解し、モデル予測を視覚化し、洞察を共有することができます。詳細については、[W&B Tablesガイド](https://docs.wandb.ai/guides/tables)をご覧ください。

```python
# wandb runを作成し、オプションで"log-dataset"ジョブタイプを指定して整理する
run = wandb.init(
    project=WANDB_PROJECT, job_type="log-dataset"
)  # configは任意です

# W&B Tableを作成し、ランダムに選んだデータセットの1000行をログする
table = wandb.Table(dataframe=trndat.sample(1000))

# W&BワークスペースにTableをログする
wandb.log({"processed_dataset": table})

# wandbのランを終了する
wandb.finish()
```

# モデリング

## XGBoostモデルをフィットさせる

車両ローン申請がデフォルトするかどうかを分類するために、XGBoostモデルをフィットさせます。

### GPUでのトレーニング
XGBoostモデルをGPUでトレーニングしたい場合は、XGBoostに渡すパラメータに以下を設定してください:

```python
"tree_method": "gpu_hist"
```

#### 1) W&Bのランを初期化する

```python
run = wandb.init(project=WANDB_PROJECT, job_type="train-model")
```

#### 2) モデルパラメータを設定し、ログする

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
    "tree_method": "hist",  # GPUでのトレーニングには `gpu_hist` を使用
}
```

XGBoostのトレーニングパラメータをW&Bランコンフィグにログする

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

## ターゲット列をシリーズとして抽出する
y_trn = trndat.loc[:, targ_var].astype(int)
y_val = valdat.loc[:, targ_var].astype(int)
```

#### 4) モデルをフィットさせ、結果をW&Bにログし、モデルをW&B Artifactsに保存する

すべてのXGBoostモデルパラメータをログするために、`WandbCallback`を使用します。詳細については、[W&B ドキュメント](https://docs.wandb.ai/guides/integrations)をご覧ください。他のライブラリでのW&Bとの統合も含まれています（LightGBMなど）。

```python
from wandb.integration.xgboost import WandbCallback

# XGBoostClassifierをWandbCallbackを使用して初期化する
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

# トレーニングおよびバリデーションの予測を取得する
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

# 追加のバリデーションメトリクスをログする
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

#### 6) ROCカーブをW&Bにログする

```python
# ROCカーブをW&Bにログする
valYpreds_2d = np.array([1 - valYpreds, valYpreds])  # W&Bは2次元配列を想定
y_val_arr = y_val.values
d = 0
while len(valYpreds_2d.T) > 10000:
    d += 1
    valYpreds_2d = valYpreds_2d[::1, ::d]
    y_val_arr = y_val_arr[::d]

run.log(
    {
        "ROC_Curve": wandb.plot.roc_curve(
            y_val_arr,
            valYpreds_2d.T,
            labels=["no_default", "loan_default"],
            classes_to_plot=[1],
        )
    }
)
```

#### W&B runを終了する

```python
run.finish()
```

これで単一のモデルをトレーニングしましたが、その性能を最適化するためにハイパーパラメータースイープを実行しましょう。

# ハイパーパラメータースイープ

Weights and Biasesでは、独自の[Sweeps機能](https://docs.wandb.ai/guides/sweeps/python-api)や[Ray Tune統合](https://docs.wandb.ai/guides/sweeps/advanced-sweeps/ray-tune)を使用してハイパーパラメータースイープを実行できます。詳細な使用方法については、[ドキュメント](https://docs.wandb.ai/guides/sweeps/python-api)をご覧ください。

[**こちらをクリック**](https://wandb.ai/morgan/credit_score_sweeps/sweeps/iuppbs45)して、このノートブックで生成された1000ランのスイープ結果をチェックしてください。

#### スイープ構成を定義する
まず、スイープするハイパーパラメーターとスイープの種類を定義します。今回はlearning_rate、gamma、min_child_weights、およびearly_stopping_roundsのランダム検索を行います。

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

#### トレーニング関数を定義する

次に、これらのハイパーパラメーターを使用してモデルをトレーニングする関数を定義します。注意すべきは、`job_type='sweep'`を使用してランを初期化している点です。これにより、必要に応じてこれらのランをメインワークスペースから簡単にフィルタリングできます。

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
            "reg_lambda": 0,  ## def: 1
            "eval_metric": ["auc", "logloss"],
            "tree_method": "hist",
        }

        # WandbCallbackを使用してXGBoostClassifierを初期化する
        xgbmodel = xgb.XGBClassifier(
            **bst_params,
            callbacks=[WandbCallback()],
            early_stopping_rounds=run.config["early_stopping_rounds"]
        )

        # モデルをトレーニングする
        xgbmodel.fit(trndat[p_vars], y_trn, eval_set=[(valdat[p_vars], y_val)])

        bstr = xgbmodel.get_booster()

        # ブースターメトリクスをログする
        run.summary["best_ntree_limit"] = bstr.best_ntree_limit

        # トレーニングおよびバリデーションの予測を取得する
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

        # 追加のバリデーションメトリクスをログする
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

#### Sweepsエージェントを実行する

```python
count = 10  # 実行するランの数
wandb.agent(sweep_id, function=train, count=count)
```

## お気に入りのMLライブラリにすでにW&Bが組み込まれています

Weights and Biasesは、お気に入りのMLおよびディープラーニングライブラリすべてと統合されています:

- Pytorch Lightning
- Keras
- Hugging Face
- JAX
- Fastai
- XGBoost
- Sci-Kit Learn
- LightGBM

