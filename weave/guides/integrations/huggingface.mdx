---
title: Hugging Face Transformers
menu:
  default:
    identifier: ja-guides-integrations-huggingface
    parent: integrations
weight: 110
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Optimize_Hugging_Face_models_with_Weights_&_Biases.ipynb" >}}

[Hugging Face Transformers](https://huggingface.co/transformers/) ライブラリは、BERTのような最先端のNLPモデルや、混合精度、勾配チェックポイントなどのトレーニング手法を簡単に使用できるようにします。[W&B integration](https://huggingface.co/transformers/main_classes/callback.html#transformers.integrations.WandbCallback)により、その使いやすさを損なうことなく、柔軟な実験管理とモデルのバージョン管理をインタラクティブな集中ダッシュボードに追加します。

## 数行で次世代のロギング

```python
os.environ["WANDB_PROJECT"] = "<my-amazing-project>"  # W&Bプロジェクトの名前を指定
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # すべてのモデルチェックポイントをログ

from transformers import TrainingArguments, Trainer

args = TrainingArguments(..., report_to="wandb")  # W&Bのログを有効化
trainer = Trainer(..., args=args)
```
{{< img src="/images/integrations/huggingface_gif.gif" alt="W&Bのインタラクティブダッシュボードで実験結果を探索" >}}

{{% alert %}}
すぐに動作するコードを試したい方は、この[Google Colab](https://wandb.me/hf)をチェックしてください。
{{% /alert %}}

## 始める： 実験をトラックする

### サインアップしてAPIキーを作成する

APIキーは、あなたのマシンをW&Bに認証します。ユーザープロフィールからAPIキーを生成できます。

{{% alert %}}
よりスムーズな方法として、[https://wandb.ai/authorize](https://wandb.ai/authorize)で直接APIキーを生成することができます。表示されたAPIキーをコピーして、パスワードマネージャーのような安全な場所に保存してください。
{{% /alert %}}

1. 右上のユーザーアイコンをクリックします。
1. **ユーザー設定**を選択し、**APIキー**セクションまでスクロールします。
1. **Reveal**をクリックします。表示されたAPIキーをコピーします。APIキーを非表示にするには、ページを再読み込みします。

### `wandb`ライブラリをインストールしてログインする

`wandb`ライブラリをローカルにインストールし、ログインするには：

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}})をAPIキーに設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb`ライブラリをインストールしてログインします。

    ```shell
    pip install wandb

    wandb login
    ```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```bash
pip install wandb
```
```python
import wandb
wandb.login()
```

{{% /tab %}}

{{% tab header="Python notebook" value="python" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

初めてW&Bを使用する場合、[**クイックスタート**]({{< relref path="/guides/quickstart.md" lang="ja" >}})をご覧になることをお勧めします。

### プロジェクトの名前を付ける

W&B Projectは、関連するRunsからログされたすべてのチャート、データ、モデルを保存する場所です。プロジェクト名をつけることで、作業を整理し、1つのプロジェクトに関するすべての情報を一ヶ所にまとめることができます。

プロジェクトにrunを追加するには、単に`WANDB_PROJECT` 環境変数をプロジェクト名に設定するだけです。`WandbCallback`は、このプロジェクト名の環境変数を拾い上げ、runを設定する際にそれを使用します。

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

```bash
WANDB_PROJECT=amazon_sentiment_analysis
```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```python
import os
os.environ["WANDB_PROJECT"]="amazon_sentiment_analysis"
```

{{% /tab %}}

{{% tab header="Python notebook" value="notebook" %}}

```notebook
%env WANDB_PROJECT=amazon_sentiment_analysis
```

{{% /tab %}}

{{< /tabpane >}}

{{% alert %}}
プロジェクト名は`Trainer`を初期化する前に設定することを確認してください。
{{% /alert %}}

プロジェクト名が指定されていない場合、プロジェクト名は`huggingface`にデフォルト設定されます。

### トレーニングRunsをW&Bにログする

これは、コード内またはコマンドラインからトレーニング引数を定義する際の**最も重要なステップ**です。`report_to`を`"wandb"`に設定することで、W&Bログを有効にします。

`TrainingArguments`の`logging_steps`引数は、トレーニング中にW&Bにトレーニングメトリクスがプッシュされる頻度を制御します。`run_name`引数を使用して、W&B内でトレーニングrunに名前を付けることもできます。

これで終了です。トレーニング中は、モデルが損失、評価メトリクス、モデルトポロジー、勾配をW&Bにログします。

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

```bash
python run_glue.py \     # Pythonスクリプトを実行
  --report_to wandb \    # W&Bにログを有効化
  --run_name bert-base-high-lr \   # W&B runの名前 (オプション)
  # その他のコマンドライン引数をここに
```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```python
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    # 他の引数やキーワード引数をここに
    report_to="wandb",  # W&Bにログを有効化
    run_name="bert-base-high-lr",  # W&B runの名前 (オプション)
    logging_steps=1,  # W&Bにログする頻度
)

trainer = Trainer(
    # 他の引数やキーワード引数をここに
    args=args,  # トレーニング引数
)

trainer.train()  # トレーニングとW&Bへのログを開始
```

{{% /tab %}}
{{< /tabpane >}}

{{% alert %}}
TensorFlowを使用していますか？ PyTorchの`Trainer`をTensorFlowの`TFTrainer`に置き換えるだけです。
{{% /alert %}}

### モデルのチェックポイントをオンにする

[Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}})を使用すると、最大100GBのモデルやデータセットを無料で保存し、その後Weights & Biasesの[Registry]({{< relref path="/guides/core/registry/" lang="ja" >}})を使用できます。Registryを使用して、モデルを登録し、それらを探索・評価したり、ステージングの準備をしたり、プロダクション環境にデプロイできます。

Hugging FaceモデルのチェックポイントをArtifactsにログするには、`WANDB_LOG_MODEL` 環境変数を以下のいずれかに設定します：

- **`checkpoint`**: [`TrainingArguments`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments)の`args.save_steps`ごとにチェックポイントをアップロードします。
- **`end`**: トレーニング終了時にモデルをアップロードします。また`load_best_model_at_end`が設定されている場合です。
- **`false`**: モデルをアップロードしません。

{{< tabpane text=true >}}

{{% tab header="Command Line" value="cli" %}}

```bash
WANDB_LOG_MODEL="checkpoint"
```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```python
import os

os.environ["WANDB_LOG_MODEL"] = "checkpoint"
```

{{% /tab %}}

{{% tab header="Python notebook" value="notebook" %}}

```notebook
%env WANDB_LOG_MODEL="checkpoint"
```

{{% /tab %}}

{{< /tabpane >}}

これ以降に初期化するすべてのTransformers `Trainer`は、モデルをW&Bプロジェクトにアップロードします。ログされたモデルチェックポイントは[Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) UIを通じて表示可能で、完全なモデルリネージを含みます（UIでのモデルチェックポイントの例はこちらをご覧ください [here](https://wandb.ai/wandb/arttest/artifacts/model/iv3_trained/5334ab69740f9dda4fed/lineage?_gl=1*yyql5q*_ga*MTQxOTYyNzExOS4xNjg0NDYyNzk1*_ga_JH1SJHJQXJ*MTY5MjMwNzI2Mi4yNjkuMS4xNjkyMzA5NjM2LjM3LjAuMA..))。

{{% alert %}}
デフォルトでは、`WANDB_LOG_MODEL`が`end`に設定されているときは`model-{run_id}`として、`WANDB_LOG_MODEL`が`checkpoint`に設定されているときは`checkpoint-{run_id}`として、モデルがW&B Artifactsに保存されます。しかし、`TrainingArguments`に[`run_name`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.run_name)を渡すと、モデルは`model-{run_name}`または`checkpoint-{run_name}`として保存されます。
{{% /alert %}}

#### W&B Registry
チェックポイントをArtifactsにログしたら、最良のモデルチェックポイントを登録して、**[Registry]({{< relref path="/guides/core/registry/" lang="ja" >}})**でチーム全体に中央集約できます。Registryを使用すると、タスクごとに最良のモデルを整理し、モデルライフサイクルを管理し、機械学習ライフサイクル全体を追跡および監査し、[オートメーション]({{< relref path="/guides/core/automations/" lang="ja" >}})ダウンストリームアクションを自動化できます。

モデルのアーティファクトをリンクするには、[Registry]({{< relref path="/guides/core/registry/" lang="ja" >}})を参照してください。

### トレーニング中に評価出力を視覚化する

トレーニングや評価中にモデル出力を視覚化することは、モデルがどのようにトレーニングされているかを理解するためにしばしば重要です。

Transformers Trainerのコールバックシステムを使用すると、モデルのテキスト生成出力や他の予測などの役立つデータをW&B Tablesにログできます。

トレーニング中にW&B Tableに評価出力をログする方法については、以下の**[カスタムログセクション]({{< relref path="#custom-logging-log-and-view-evaluation-samples-during-training" lang="ja" >}})**をご覧ください:

{{< img src="/images/integrations/huggingface_eval_tables.png" alt="評価出力を含むW&B Tableを表示" >}}

### W&B Runを終了させる（ノートブックのみ）

トレーニングがPythonスクリプトでカプセル化されている場合、スクリプトが終了するとW&B runも終了します。

JupyterまたはGoogle Colabノートブックを使用している場合は、トレーニングが終了したことを`wandb.finish()`を呼び出して知らせる必要があります。

```python
trainer.train()  # トレーニングとW&Bへのログを開始

# トレーニング後の分析、テスト、他のログ済みコード

wandb.finish()
```

### 結果を視覚化する

トレーニング結果をログしたら、[W&B Dashboard]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}})で結果を動的に探索できます。複数のrunを一度に比較したり、興味深い知見にズームインしたり、柔軟でインタラクティブな可視化を用いて複雑なデータから洞察を引き出すのが簡単です。

## 高度な機能とFAQ

### 最良のモデルを保存する方法は？

`Trainer`に`load_best_model_at_end=True`の`TrainingArguments`を渡すと、W&Bは最良のパフォーマンスを示すモデルチェックポイントをアーティファクトに保存します。

モデルチェックポイントをアーティファクトとして保存すれば、それらを[Registry]({{< relref path="/guides/core/registry/" lang="ja" >}})に昇格させることができます。Registryでは以下のことが可能です：
- MLタスクによって最良のモデルバージョンを整理する。
- モデルを集約してチームと共有する。
- モデルをステージングしてプロダクションに展開するか、さらに評価するためにブックマークする。
- 下流のCI/CDプロセスをトリガーする。

### 保存したモデルをロードするには？

`WANDB_LOG_MODEL`でW&B Artifactsにモデルを保存した場合、追加トレーニングや推論のためにモデルウェイトをダウンロードできます。同じHugging Faceアーキテクチャーにモデルを読み戻すだけです。

```python
# 新しいrunを作成
with wandb.init(project="amazon_sentiment_analysis") as run:
    # アーティファクトの名前とバージョンを指定
    my_model_name = "model-bert-base-high-lr:latest"
    my_model_artifact = run.use_artifact(my_model_name)

    # フォルダーにモデルウェイトをダウンロードし、パスを返す
    model_dir = my_model_artifact.download()

    # 同じモデルクラスを使用して、そのフォルダーからHugging Faceモデルをロード
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=num_labels
    )

    # 追加のトレーニングを行うか、推論を実行
```

### チェックポイントからトレーニングを再開するには？

`WANDB_LOG_MODEL='checkpoint'`を設定していた場合、`model_dir`を`TrainingArguments`の`model_name_or_path`引数として使用し、`Trainer`に`resume_from_checkpoint=True`を渡すことでトレーニングを再開できます。

```python
last_run_id = "xxxxxxxx"  # wandb workspaceからrun_idを取得

# run_idからwandb runを再開
with wandb.init(
    project=os.environ["WANDB_PROJECT"],
    id=last_run_id,
    resume="must",
) as run:
    # アーティファクトをrunに接続
    my_checkpoint_name = f"checkpoint-{last_run_id}:latest"
    my_checkpoint_artifact = run.use_artifact(my_model_name)

    # フォルダーにチェックポイントをダウンロードし、パスを返す
    checkpoint_dir = my_checkpoint_artifact.download()

    # モデルとトレーナーを再初期化
    model = AutoModelForSequenceClassification.from_pretrained(
        "<model_name>", num_labels=num_labels
    )
    # 素晴らしいトレーニング引数をここに
    training_args = TrainingArguments()

    trainer = Trainer(model=model, args=training_args)

    # チェックポイントディレクトリを使用してトレーニングをチェックポイントから再開することを確かにする
    trainer.train(resume_from_checkpoint=checkpoint_dir)
```

### トレーニング中に評価サンプルをログして表示するには？

Transformers `Trainer`を介してW&Bにログすることは、Transformersライブラリの[`WandbCallback`](https://huggingface.co/transformers/main_classes/callback.html#transformers.integrations.WandbCallback)によって処理されます。Hugging Faceのログをカスタマイズする必要がある場合は、`WandbCallback`をサブクラス化し、Trainerクラスから追加のメソッドを利用する追加機能を追加することにより、このコールバックを変更できます。

以下は、HF Trainerにこの新しいコールバックを追加する際の一般的なパターンであり、さらに下にはW&B Tableに評価出力をログするコード完備の例があります：

```python
# 通常通りTrainerをインスタンス化
trainer = Trainer()

# Trainerオブジェクトを渡して新しいログコールバックをインスタンス化
evals_callback = WandbEvalsCallback(trainer, tokenizer, ...)

# Trainerにコールバックを追加
trainer.add_callback(evals_callback)

# 通常通りTrainerトレーニングを開始
trainer.train()
```

#### トレーニング中に評価サンプルを表示

以下のセクションでは、`WandbCallback`をカスタマイズして、モデルの予測を実行し、トレーニング中にW&B Tableに評価サンプルをログする方法を示します。`on_evaluate`メソッドを使用して`eval_steps`ごとにログします。

ここでは、トークナイザーを使用してモデル出力から予測とラベルをデコードするための`decode_predictions`関数を書いています。

その後、予測とラベルからpandas DataFrameを作成し、DataFrameに`epoch`列を追加します。

最後に、DataFrameから`wandb.Table`を作成し、それをwandbにログします。
さらに、`freq`エポックごとに予測をログすることで、ログの頻度を制御できます。

**注意**: 通常の`WandbCallback`とは異なり、このカスタムコールバックは`Trainer`の初期化時ではなく、`Trainer`がインスタンス化された後でトレーナーに追加する必要があります。これは、`Trainer`インスタンスが初期化中にコールバックに渡されるためです。

```python
from transformers.integrations import WandbCallback
import pandas as pd


def decode_predictions(tokenizer, predictions):
    labels = tokenizer.batch_decode(predictions.label_ids)
    logits = predictions.predictions.argmax(axis=-1)
    prediction_text = tokenizer.batch_decode(logits)
    return {"labels": labels, "predictions": prediction_text}


class WandbPredictionProgressCallback(WandbCallback):
    """トレーニング中にモデルの予測をログするカスタムWandbCallback。

    このコールバックは、トレーニング中の各ログステップでモデルの予測とラベルをwandb.Tableにログします。トレーニングの進行に応じたモデルの予測を視覚化することができます。

    Attributes:
        trainer (Trainer): Hugging Face Trainerインスタンス。
        tokenizer (AutoTokenizer): モデルに関連付けられたトークナイザー。
        sample_dataset (Dataset): 予測を生成するための
          検証データセットのサブセット。
        num_samples (int, optional): 検証データセットから選択するサンプルの数。
          デフォルトは100。
        freq (int, optional): ログの頻度。デフォルトは2。
    """

    def __init__(self, trainer, tokenizer, val_dataset, num_samples=100, freq=2):
        """WandbPredictionProgressCallbackインスタンスを初期化します。

        Args:
            trainer (Trainer): Hugging Face Trainerインスタンス。
            tokenizer (AutoTokenizer): モデルに関連付けられたトークナイザー。
            val_dataset (Dataset): 検証データセット。
            num_samples (int, optional): 予測を生成するために
              検証データセットから選択するサンプルの数。
              デフォルトは100。
            freq (int, optional): ログの頻度。デフォルトは2。
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset.select(range(num_samples))
        self.freq = freq

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        # `freq`エポックごとに予測をログすることにより、ログの頻度を制御
        if state.epoch % self.freq == 0:
            # 予測を生成
            predictions = self.trainer.predict(self.sample_dataset)
            # 予測とラベルをデコード
            predictions = decode_predictions(self.tokenizer, predictions)
            # 予測をwandb.Tableに追加
            predictions_df = pd.DataFrame(predictions)
            predictions_df["epoch"] = state.epoch
            records_table = self._wandb.Table(dataframe=predictions_df)
            # テーブルをwandbにログ
            self._wandb.log({"sample_predictions": records_table})


# まずはTrainerをインスタンス化
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
)

# WandbPredictionProgressCallbackをインスタンス化
progress_callback = WandbPredictionProgressCallback(
    trainer=trainer,
    tokenizer=tokenizer,
    val_dataset=lm_dataset["validation"],
    num_samples=10,
    freq=2,
)

# コールバックをトレーナーに追加
trainer.add_callback(progress_callback)
```

詳細な例については、この[colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Custom_Progress_Callback.ipynb)を参照してください。

### 利用可能な追加のW&B設定は？

`Trainer`でログされる内容のさらなる設定は、環境変数を設定することで可能です。W&B環境変数の完全なリストは[こちらにあります]({{< relref path="/guides/hosting/env-vars/" lang="ja" >}})。

| 環境変数 | 使用法                                                                                                                                                                                                                                                                                                      |
| -------------------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `WANDB_PROJECT`      | プロジェクト名を付けます（デフォルトは`huggingface`）                                                                                                                                                                                                                                           |
| `WANDB_LOG_MODEL`    | <p>モデルチェックポイントをW&Bアーティファクトとしてログします（デフォルトは`false`） </p><ul><li><code>false</code>（デフォルト）：モデルチェックポイントは行われません </li><li><code>checkpoint</code>：args.save_stepsごとにチェックポイントがアップロードされます（TrainerのTrainingArgumentsで設定） </li><li><code>end</code>：トレーニングの終了時に最終モデルチェックポイントがアップロードされます。</li></ul>                                                                                                                                                                                                                    |
| `WANDB_WATCH`        | <p>モデルの勾配、パラメータ、またはそのいずれもログするかどうかを設定します</p><ul><li><code>false</code>（デフォルト）：勾配やパラメータのログは行わない </li><li><code>gradients</code>：勾配のヒストグラムをログ </li><li><code>all</code>：勾配とパラメータのヒストグラムをログ</li></ul> |
| `WANDB_DISABLED`     | `true`に設定すると、ログが完全にオフになります（デフォルトは`false`）                                                                                                                                                                                                                              |
| `WANDB_SILENT`       | `true`に設定すると、wandbによって印刷される出力が消音されます（デフォルトは`false`）                                                                                                                                                                                                               |

{{< tabpane text=true >}}

{{% tab header="Command Line" value="cli" %}}

```bash
WANDB_WATCH=all
WANDB_SILENT=true
```

{{% /tab %}}

{{% tab header="Notebook" value="notebook" %}}

```notebook
%env WANDB_WATCH=all
%env WANDB_SILENT=true
```

{{% /tab %}}

{{< /tabpane >}}

### `wandb.init`をカスタマイズする方法は？

`Trainer`が使用する`WandbCallback`は、`Trainer`が初期化される際に内部的に`wandb.init`を呼び出します。代わりに、`Trainer`が初期化される前に`wandb.init`を手動で呼び出してrunを設定することもできます。これにより、W&Bのrun設定を完全にコントロールできます。

以下は、`init`に何を渡すかの例です。`wandb.init`の使用方法の詳細については、[リファレンスドキュメントを参照してください]({{< relref path="/ref/python/init.md" lang="ja" >}})。

```python
wandb.init(
    project="amazon_sentiment_analysis",
    name="bert-base-high-lr",
    tags=["baseline", "high-lr"],
    group="bert",
)
```

## 追加のリソース

以下は、6つのTransformersとW&Bに関連する記事で楽しめるかもしれないものです。

<details>

<summary>Hugging Face Transformersのハイパーパラメータ最適化</summary>

* Hugging Face Transformersのハイパーパラメータ最適化のための3つの戦略：グリッド検索、ベイズ最適化、population based trainingが比較されています。
* Hugging Face transformersの標準的なベースラインモデルを使用し、SuperGLUEベンチマークからRTEデータセットを使用してファインチューニングしたいと考えています。
* 結果は、population based trainingがHugging Face transformerモデルのハイパーパラメータ最適化に最も効果的なアプローチであることを示しています。

詳細なレポートは[こちら](https://wandb.ai/amogkam/transformers/reports/Hyperparameter-Optimization-for-Hugging-Face-Transformers--VmlldzoyMTc2ODI)をご覧ください。
</details>

<details>

<summary>Hugging Tweets: ツイートを生成するモデルをトレーニング</summary>

* 記事では、著者が任意の人のツイートを5分で再学習するようにGPT2 HuggingFace Transformerモデルをファインチューニングする方法を実演します。
* モデルは以下のパイプラインを使用します：ツイートのダウンロード、データセットの最適化、初期実験、ユーザー間の損失の比較、モデルのファインチューニング。

詳細なレポートは[こちら](https://wandb.ai/wandb/huggingtweets/reports/HuggingTweets-Train-a-Model-to-Generate-Tweets--VmlldzoxMTY5MjI)をご覧ください。
</details>

<details>

<summary>Hugging Face BERTとWBによる文の分類</summary>

* この記事では、自然言語処理の最近のブレークスルーの力を活用した文分類器の構築について説明します。NLPへの転移学習の適用に焦点を当てています。
* 文法的に正しいかどうかをラベル付けした文のセットである、単一文分類用のThe Corpus of Linguistic Acceptability (CoLA) データセットを使用します。このデータセットは2018年5月に初めて公開されました。
* GoogleのBERTを使用して、最小限の努力で様々なNLPタスクで高性能なモデルを作成します。

詳細なレポートは[こちら](https://wandb.ai/cayush/bert-finetuning/reports/Sentence-Classification-With-Huggingface-BERT-and-W-B--Vmlldzo4MDMwNA)をご覧ください。
</details>

<details>

<summary>Hugging Faceモデルパフォーマンスをトラックするためのステップバイステップガイド</summary>

* W&Bと Hugging Face transformers を使って、BERT の97%の精度を維持しつつ、40%小さいTrasformerであるDistilBERTをGLUEベンチマークでトレーニングします。
* GLUEベンチマークは、NLPモデルをトレーニングするための9つのデータセットとタスクのコレクションです。

詳細なレポートは[こちら](https://wandb.ai/jxmorris12/huggingface-demo/reports/A-Step-by-Step-Guide-to-Tracking-HuggingFace-Model-Performance--VmlldzoxMDE2MTU)をご覧ください。
</details>

<details>

<summary>HuggingFaceにおけるEarly Stoppingの例</summary>

* Early Stopping正則化を使用して、Hugging Face Transformerをファインチューニングすることは、PyTorchやTensorFlowでネイティブに実行できます。
* TensorFlowでEarlyStoppingコールバックを使用する方法は、`tf.keras.callbacks.EarlyStopping` コールバックを使って簡単にできます。
* PyTorchでは、オフの早期停止メソッドはありませんが、GitHub Gistで利用可能な早期停止フックがあります。

詳細なレポートは[こちら](https://wandb.ai/ayush-thakur/huggingface/reports/Early-Stopping-in-HuggingFace-Examples--Vmlldzo0MzE2MTM)をご覧ください。
</details>

<details>

<summary>カスタムデータセットでHugging Face Transformersをファインチューニングする方法</summary>

カスタムIMDBデータセットでセンチメント分析（二項分類）のためにDistilBERT transformerをファインチューニングします。

詳細なレポートは[こちら](https://wandb.ai/ayush-thakur/huggingface/reports/How-to-Fine-Tune-HuggingFace-Transformers-on-a-Custom-Dataset--Vmlldzo0MzQ2MDc)をご覧ください。
</details>

## ヘルプを受けたり、機能をリクエストする

Hugging Face W&Bインテグレーションに関する問題、質問、または機能のリクエストについては、[Hugging Faceフォーラムのこのスレッド](https://discuss.huggingface.co/t/logging-experiment-tracking-with-w-b/498)に投稿するか、Hugging Face[Transformers GitHubリポジトリ](https://github.com/huggingface/transformers)で問題を開いてください。