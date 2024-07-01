---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# Hugging Face Transformers

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Optimize_Hugging_Face_models_with_Weights_&_Biases.ipynb"></CTAButtons>

[Hugging Face Transformers](https://huggingface.co/transformers/) ライブラリは、最先端のNLPモデル（例えばBERT）や、混合精度や勾配チェックポイントといったトレーニング技術を簡単に利用できるようにします。[W&Bインテグレーション](https://huggingface.co/transformers/main\_classes/callback.html#transformers.integrations.WandbCallback) は、インタラクティブな集中ダッシュボードでリッチで柔軟な実験管理とモデルバージョン管理を提供し、その使いやすさを損ないません。

## 🤗 数行で次のレベルのログ記録

```python
os.environ["WANDB_PROJECT"] = "<my-amazing-project>"  # W&Bプロジェクトの名前を指定
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # すべてのモデルチェックポイントをログ

from transformers import TrainingArguments, Trainer

args = TrainingArguments(..., report_to="wandb")  # W&Bログを有効化
trainer = Trainer(..., args=args)
```
![W&Bインタラクティブダッシュボードで実験結果を探索](@site/static/images/integrations/huggingface_gif.gif)

:::info
すぐに作業コードに取り掛かりたい場合は、この[Google Colab](https://wandb.me/hf)をチェックしてください。
:::

## はじめに: 実験をトラック

### 1) サインアップ、`wandb`ライブラリのインストール、ログイン

a) [**サインアップ**](https://wandb.ai/site)して無料アカウントを作成

b) `wandb`ライブラリをpipでインストール

c) トレーニングスクリプトにログインするには、www.wandb.aiでサインインして、[**承認ページ**](https://wandb.ai/authorize)でAPIキーを見つける必要があります。

Weights and Biases を初めて使用する場合は、[**クイックスタート**](../../quickstart.md)をチェックすることをお勧めします。

<Tabs
  defaultValue="cli"
  values={[
    {label: 'Python', value: 'python'},
    {label: 'Command Line', value: 'cli'},
  ]}>
  <TabItem value="cli">

```shell
pip install wandb

wandb login
```

  </TabItem>
  <TabItem value="python">

```notebook
!pip install wandb

import wandb
wandb.login()
```

  </TabItem>
</Tabs>

### 2) プロジェクトの名前を付ける

[Project](../app/pages/project-page.md) は、関連するRunsからログされたすべてのチャート、データ、モデルが保存される場所です。プロジェクトに名前を付けることで、作業を整理し、単一のプロジェクトに関するすべての情報を一か所にまとめることができます。

プロジェクトにRunを追加するには、簡単に`WANDB_PROJECT`環境変数をプロジェクトの名前に設定します。`WandbCallback`はこのプロジェクト名の環境変数を検出し、それを使ってRunを設定します。

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python', value: 'python'},
    {label: 'Command Line', value: 'cli'},
    {label: 'Notebook', value: 'notebook'}
  ]}>
  <TabItem value="cli">

```bash
WANDB_PROJECT=amazon_sentiment_analysis
```

  </TabItem>
  <TabItem value="notebook">

```notebook
%env WANDB_PROJECT=amazon_sentiment_analysis
```

  </TabItem>
  <TabItem value="python">

```notebook
import os
os.environ["WANDB_PROJECT"]="amazon_sentiment_analysis"
```

  </TabItem>
</Tabs>


:::info
`Trainer`を初期化する前にプロジェクト名を設定してください。
:::

プロジェクト名が指定されていない場合、プロジェクト名はデフォルトで「huggingface」になります。

### 3) トレーニングのRunsをW&Bにログする

これが**最も重要なステップ**です：コード内またはコマンドラインから`Trainer`トレーニング引数を定義する際に、W&Bへのログを有効にするために、`report_to`を`"wandb"`に設定します。

`TrainingArguments`の`logging_steps`引数は、トレーニング中にトレーニングメトリクスをW&Bにプッシュする頻度をコントロールします。また、`run_name`引数を使用して、W&B内でトレーニングRunの名前を付けることもできます。

これで完了です！トレーニング中にモデルの損失、評価メトリクス、モデルトポロジー、勾配がWeights & Biasesにログされます。

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python', value: 'python'},
    {label: 'Command Line', value: 'cli'},
  ]}>
  <TabItem value="cli">

```bash
python run_glue.py \     # Pythonスクリプトを実行
  --report_to wandb \    # W&Bへのログを有効化
  --run_name bert-base-high-lr \   # W&B Runの名前（オプション）
  # 他のコマンドライン引数
```

  </TabItem>
  <TabItem value="python">

```python
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    # 他の引数とキーワード引数
    report_to="wandb",  # W&Bへのログを有効化
    run_name="bert-base-high-lr",  # W&B Runの名前（オプション）
    logging_steps=1,  # W&Bへのログ頻度
)

trainer = Trainer(
    # 他の引数とキーワード引数
    args=args,  # あなたのトレーニング引数
)

trainer.train()  # トレーニングとW&Bへのログを開始
```

  </TabItem>
</Tabs>


:::info
TensorFlowを使用していますか？PyTorchの`Trainer`をTensorFlowの`TFTrainer`に置き換えてください。
:::

### 4) モデルのチェックポイントを有効にする

Weights & Biasesの[Artifacts](../artifacts)を使用すると、無料で最大100GBのモデルとデータセットを保存し、Weights & Biases[Model Registry](../model_registry)を使用してモデルを登録し、プロダクション環境でのステージングやデプロイメントの準備をすることができます。

 Hugging Face モデルのチェックポイントをArtifactsにログするには、環境変数`WANDB_LOG_MODEL`を`end`、`checkpoint`、または`false`のいずれかに設定します：

-  **`checkpoint`**: [`TrainingArguments`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments) の `args.save_steps` ごとにチェックポイントがアップロードされます。
- **`end`**: トレーニングの最後にモデルがアップロードされます。

トレーニングの最後に最適なモデルをアップロードするには、`WANDB_LOG_MODEL`と`load_best_model_at_end`を使用します。

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python', value: 'python'},
    {label: 'Command Line', value: 'cli'},
    {label: 'Notebook', value: 'notebook'},
  ]}>

  <TabItem value="python">

```python
import os

os.environ["WANDB_LOG_MODEL"] = "checkpoint"
```

  </TabItem>
  <TabItem value="cli">

```bash
WANDB_LOG_MODEL="checkpoint"
```

  </TabItem>
  <TabItem value="notebook">

```notebook
%env WANDB_LOG_MODEL="checkpoint"
```

  </TabItem>
</Tabs>


今後初期化するすべてのTransformers`Trainer`は、モデルをW&Bプロジェクトにアップロードします。ログされたモデルのチェックポイントは[Artifacts](../artifacts)UIで表示でき、完全なモデルリネージを含みます（UIでのモデルチェックポイントの例は[こちら](https://wandb.ai/wandb/arttest/artifacts/model/iv3_trained/5334ab69740f9dda4fed/lineage?_gl=1*yyql5q*_ga*MTQxOTYyNzExOS4xNjg0NDYyNzk1*_ga_JH1SJHJQXJ*MTY5MjMwNzI2Mi4yNjkuMS4xNjkyMzA5NjM2LjM3LjAuMA..)）。

:::info
デフォルトでは、`WANDB_LOG_MODEL`が`end`に設定されている場合、モデルは`model-{run_id}`という名前で、`WANDB_LOG_MODEL`が`checkpoint`に設定されている場合は`checkpoint-{run_id}`という名前でW&B Artifactsに保存されます。
ただし、`TrainingArguments`で[`run_name`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.run_name)を渡した場合、モデルは`model-{run_name}`または`checkpoint-{run_name}`という名前で保存されます。
:::

#### W&B Model Registry
チェックポイントがArtifactsにログされた後、Weights & Biasesの**[Model Registry](../model_registry)**を使用してチーム全体でモデルを登録し、中心的に管理することができます。ここでは、タスクごとに最適なモデルを整理し、モデルのライフサイクルを管理し、MLライフサイクル全体での追跡と監査を容易にし、[自動化](https://docs.wandb.ai/guides/models/automation)された下流のアクションをWebhooksやジョブで実行できます。

Model RegistryにモデルArtifactをリンクする方法については、[Model Registry](../model_registry)のドキュメントを参照してください。

### 5) トレーニング中に評価出力を可視化

トレーニングまたは評価中にモデルの出力を可視化することは、モデルのトレーニング状況を正確に理解するために重要です。

Transformers Trainerのコールバックシステムを使用すると、テキスト生成出力やその他の予測をW&B Tablesにログすることができるなど、追加の有用なデータをW&Bにログできます。

トレーニング中に評価出力をログする方法については、以下の**[カスタムログセクション](#custom-logging-log-and-view-evaluation-samples-during-training)**をご参照ください。ここでは、次のようなW&B Tableに評価サンプルをログする完全ガイドを提供しています。

![評価出力を表示するW&Bテーブル](/images/integrations/huggingface_eval_tables.png)

### 6) W&B Runを終了する（ノートブックのみ）

トレーニングがPythonスクリプト内にカプセル化されている場合、スクリプトが終了するとW&B runも終了します。

JupyterやGoogle Colabノートブックを使用している場合は、トレーニングが完了したことを`wandb.finish()`を呼び出して知らせる必要があります。

```python
trainer.train()  # トレーニングとW&Bへのログを開始

# トレーニング後の分析、テスト、その他のログコード

wandb.finish()
```

### 7) 結果を可視化する

トレーニング結果をログしたら、[W&Bダッシュボード](../track/app.md)で結果を動的に探索できます。数十のRunsを一度に比較したり、興味深い発見をズームインしたりしながら、柔軟でインタラクティブなビジュアライゼーションで複雑なデータからインサイトを引き出すことが簡単です。

## 高度な機能とFAQ

### 最適なモデルを保存するには？
`Trainer`に渡される`TrainingArguments`で`load_best_model_at_end=True`が設定されている場合、W&Bは最適なモデルチェックポイントをArtifactsに保存します。

最適なモデルバージョンをチーム全体で一元管理し、MLタスクごとに整理し、プロダクションステージングのために用意したり、さらに評価のためにブックマークしたり、下流のモデルCI/CDプロセスを開始するには、モデルのチェックポイントをArtifactsに保存してください。Artifactsにログされた後、これらのチェックポイントは[Model Registry](../model_registry/intro.md)に昇格できます。

### 保存されたモデルをロードする

`WANDB_LOG_MODEL`でモデルをW&B Artifactsに保存した場合、追加のトレーニングや推論のためにモデルの重みをダウンロードできます。同じHugging Faceアーキテクチャに重みをロードし直します。

```python
# 新しいrunを作成
with wandb.init(project="amazon_sentiment_analysis") as run:
    # Artifactの名前とバージョンを指定
    my_model_name = "model-bert-base-high-lr:latest"
    my_model_artifact = run.use_artifact(my_model_name)

    # モデルの重みをフォルダにダウンロードし、パスを返す
    model_dir = my_model_artifact.download()

    # 同じモデルクラスを使用して、そのフォルダからHugging Faceモデルをロード
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=num_labels
    )

    # 追加トレーニングまたは推論を行う
```

### チェックポイントからトレーニングを再開する
`WANDB_LOG_MODEL='checkpoint'`を設定した場合、`model_dir`を`TrainingArguments`の`model_name_or_path`引数として使用し、`Trainer`に`resume_from_checkpoint=True`を設定してトレーニングを再開することができます。

```python
last_run_id = "xxxxxxxx"  # あなたのwandbワークスペースからrun_idを取得

# run_idからW&B runを再開
with wandb.init(
    project=os.environ["WANDB_PROJECT"],
    id=last_run_id,
    resume="must",
) as run:
    # RunにArtifactを接続
    my_checkpoint_name = f"checkpoint-{last_run_id}:latest"
    my_checkpoint_artifact = run.use_artifact(my_model_name)

    # チェックポイントをフォルダにダウンロードし、パスを返す
    checkpoint_dir = my_checkpoint_artifact.download()

    # モデルとトレーナーを再初期化
    model = AutoModelForSequenceClassification.from_pretrained(
        "<model_name>", num_labels=num_labels
    )
    # あなたの素晴らしいトレーニング引数
    training_args = TrainingArguments()

    trainer = Trainer(model=model, args=training_args)

    # チェックポイントディレクトリを使用してトレーニングを再開
    trainer.train(resume_from_checkpoint=checkpoint_dir)
```

### カスタムログ: トレーニング中に評価サンプルをログ・表示

Transformers`Trainer`経由でWeights & Biasesにログを記録するのは、Transformersライブラリの[`WandbCallback`](https://huggingface.co/transformers/main\_classes/callback.html#transformers.integrations.WandbCallback)によって処理されます。Hugging Faceのログをカスタマイズする必要がある場合は、`WandbCallback`をサブクラス化し、Trainerクラスの追加メソッドを活用して機能を追加することでこのコールバックを修正できます。

以下は、新しいコールバックをHF Trainerに追加する一般的なパターンです。さらに下のコード例には、トレーニング中に評価出力をW&B Tableにログする完全な例が示されています。

```python
# 通常通りTrainerをインスタンス化
trainer = Trainer()

# Trainerオブジェクトを渡して、新しいログコールバックをインスタンス化
evals_callback = WandbEvalsCallback(trainer, tokenizer, ...)

# コールバックをTrainerに追加
trainer.add_callback(evals_callback)

# 通常通りTrainerトレーニングを開始
trainer.train()
```

#### トレーニング中に評価サンプルを表示

次のセクションでは、`WandbCallback`をカスタマイズしてモデル予測を実行し、トレーニング中に評価サンプルをW&B Tableにログする方法を示します。`on_evaluate`メソッドを使用して`eval_steps`ごとに評価を行います。

ここでは、`tokenizer`を使用してモデル出力から予測とラベルをデコードする`decode_predictions`関数を作成しました。

次に、予測とラベルからpandas DataFrameを作成し、DataFrameに`epoch`列を追加します。

最後に、DataFrameから`wandb.Table`を作成し、それをwandbにログします。
また、ログの頻度を調整するために、`freq`エポックごとに予測をログすることができます。

**注意**: 通常の`WandbCallback`とは異なり、このカスタムコールバックは`Trainer`がインスタンス化された後に追加する必要があります。これは、コールバックの初期化中に`Trainer`インスタンスがコールバックに渡されるためです。

```python
from transformers.integrations import WandbCallback
import pandas as pd


def decode_predictions(tokenizer, predictions):
    labels = tokenizer.batch_decode(predictions.label_ids)
    logits = predictions.predictions.argmax(axis=-1)
    prediction_text = tokenizer.batch_decode(logits)
    return {"labels": labels, "predictions": prediction_text}


class WandbPredictionProgressCallback(WandbCallback):
    """カスタムWandbCallback: トレーニング中のモデル予測をログ.

    このコールバックは、トレーニング中の各ログ記録ステップでモデルの予測とラベルをwandb.Tableに記録します。
    トレーニングが進むにつれてモデルの予測を可視化できます。

    属性:
        trainer (Trainer): Hugging Face Trainerインスタンス
        tokenizer (AutoTokenizer): モデルに関連付けられたトークナイザー
        sample_dataset (Dataset): 予測を生成するための検証データセットのサブセット
        num_samples (int, optional): 予測を生成するための検証データセットから選択するサンプル数。デフォルトは100.
        freq (int, optional): ログの頻度。デフォルトは2.
    """

    def __init__(self, trainer, tokenizer, val_dataset,
                 num_samples=100, freq=2):
        """WandbPredictionProgressCallbackの初期化

        引数:
            trainer (Trainer): Hugging Face Trainerインスタンス
            tokenizer (AutoTokenizer): モデルに関連付けられたトークナイザー
            val_dataset (Dataset): 検証データセット
            num_samples (int, optional): 予測を生成するための検証データセットから選択するサンプル数。デフォルトは100.
            freq (int, optional): ログの頻度。デフォルトは2.
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset.select(range(num_samples))
        self.freq = freq

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        # 予測を`freq`エポックごとにログ
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


# まずTrainerをインスタンス化
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

# コールバックをTrainerに追加
trainer.add_callback(progress_callback)
```

詳細な例については、この[colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Custom_Progress_Callback.ipynb)を参照してください。


### 追加のW&B設定

`Trainer`を使用してログされる内容のさらなる設定は、環境変数を設定することで可能です。W&B環境変数の完全なリストは[こちら](https://docs.wandb.ai/library/environment-variables)にあります。

| 環境変数                | 使用方法                                                                                                                                                                                                                                                                           |
| ----------------------- |------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `WANDB_PROJECT`         | プロジェクトに名前を付けます（デフォルトは`huggingface`）                                                                                                                                                                                                                          |
| `WANDB_LOG_MODEL`       | <p>モデルのチェックポイントをW&B Artifactとしてログする（デフォルトは`false`） </p><ul><li><code>false</code>（デフォルト）: モデルのチェックポイントをログしない </li><li><code>checkpoint</code>: args.save_stepsごとにチェックポイントがアップロードされます（TrainerのTrainingArgumentsで設定）。 </li><li><code>end</code>: トレーニングの最後に最終モデルのチェックポイントがアップロードされます。</li></ul>                                |
| `WANDB_WATCH`           | <p>モデルの勾配、パラメーター、またはどちらもログするかどうかを設定します</p><ul><li><code>false</code>（デフォルト）: 勾配やパラメーターをログしない </li><li><code>gradients</code>: 勾配のヒストグラムをログ </li><li><code>all</code>: 勾配とパラメーターのヒストグラムをログ</li></ul>                                                |
| `WANDB_DISABLED`        | `true`に設定するとログを完全に無効化します（デフォルトは`false`）                                                                                                                                                                                                                   |
| `WANDB_SILENT`          | `true`に設定するとwandbから出力される出力をサイレントにします（デフォルトは`false`）                                                                                                                                                                                                |

<Tabs
  defaultValue="cli"
  values={[
    {label: 'Command Line', value: 'cli'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="cli">

```bash
WANDB_WATCH=all
WANDB_SILENT=true
```

  </TabItem>
  <TabItem value="notebook">

```notebook
%env WANDB_WATCH=all
%env WANDB_SILENT=true
```

  </TabItem>
</Tabs>

### `wandb.init` をカスタマイズ

`Trainer`が使用する`WandbCallback`は、`Trainer`が初期化されるときに内部で`wandb.init`を呼び出します。`Trainer`を初期化する前に手動で`wandb.init`を呼び出すことで、W&B runの設定を完全にカスタマイズできます。

以下は、`init`に渡す可能性がある内容の例です。`wandb.init`の使用方法についての詳細は[リファレンスドキュメント](../../ref/python/init.md)を参照してください。

```python
wandb.init(
    project="amazon_sentiment_analysis",
    name="bert-base-high-lr",
    tags=["baseline", "high-lr"],
    group="bert",
)
```

## 注目の記事

以下は、TransformersとW&Bに関連する記事6選です。

<details>

<summary>Hugging Face Transformers のハイパーパラメーター最適化</summary>

* Hugging Face Transformers のハイパーパラメーター最適化の三つの戦略 - グリッド検索、ベイズ最適化、Population Based Training が比較されています。
* Hugging Face transformers の標準的な未ケース BERT モデルを使用し、SuperGLUE ベンチマークの RTE データセットでファインチューンします。
* 結果は、Population Based Training が Hugging Face transformer モデルのハイパーパラメーター最適化に最も効果的なアプローチであることを示しています。

完全なレポートは[こちら](https://wandb.ai/amogkam/transformers/reports/Hyperparameter-Optimization-for-Hugging-Face-Transformers--VmlldzoyMTc2ODI)。
</details>

<details>

<summary>Hugging Tweets: ツイートを生成するモデルをトレーニング</summary>

* 記事では、事前学習済みGPT2 HuggingFace Transformerモデルを使用し、誰でも彼のツイートに対してわずか5分でファインチューンする方法が示されています。
* モデルは次のパイプラインを使用します：ツイートのダウンロード、データセットの最適化、初期実験、ユーザー間の損失の比較、モデルのファインチューン。

完全なレポートは[こちら](https://wandb.ai/wandb/huggingtweets/reports/HuggingTweets-Train-a-Model-to-Generate-Tweets--VmlldzoxMTY5MjI)。
</details>

<details>

<summary>Hugging Face BERT と WBを使用した文分類</summary>

* 記事では、最近のNLPのブレークスルーを活用し、転移学習をNLPに応用する文分類器を構築します。
* 単一文分類用のCoLA（Corpus of Linguistic Acceptability）データセットを使用します。これは、文が文法的に正しいかどうかをラベル付けしたもので、最初は2018年5月に公開されました。
* GoogleのBERTを使用して、最小限の労力で幅広いNLPタスクに対して高性能なモデルを作成します。

完全なレポートは[こちら](https://wandb.ai/cayush/bert-finetuning/reports/Sentence-Classification-With-Huggingface-BERT-and-W-B--Vmlldzo4MDMwNA)。
</details>

<details>

<summary>Hugging Face モデルのパフォーマンスをトラックするためのステップバイステップガイド</summary>

* Weights & BiasesとHugging Face transformersを使用して、BERTよりも40％小さいがBERTの精度の97％を保つDistilBERTをGLUEベンチマークでトレーニングします。
* GLUEベンチマークは、NLPモデルのトレーニングのための9つのデータセットとタスクのコレクションです。

完全なレポートは[こちら](https://wandb.ai/jxmorris12/huggingface-demo/reports/A-Step-by-Step-Guide-to-Tracking-HuggingFace-Model-Performance--VmlldzoxMDE2MTU)。
</details>

<details>

<summary>HuggingFaceでのアーリーストッピング - 例</summary>

* アーリーストッピング正規化を使用してHugging Face Transformerをファインチューンするには、PyTorchまたはTensorFlowでネイティブに行うことができます。
* TensorFlowでのアーリーストッピングコールバックを使用することは、`tf.keras.callbacks.EarlyStopping`コールバックを使用することで簡単です。
* PyTorchにはオフ・ザ・シェルフのアーリーストッピングメソッドはありませんが、GitHubのGistで動作するアーリーストッピングフックがあります。

完全なレポートは[こちら](https://wandb.ai/ayush-thakur/huggingface/reports/Early-Stopping-in-HuggingFace-Examples--Vmlldzo0MzE2MTM)。
</details>

<details>

<summary>カスタムデータセットでHugging Face Transformersをファインチューンする方法</summary>

カスタムIMDBデータセットでセンチメント分析（二値分類）のためにDistilBERTトランスフォーマーをファインチューンします。

完全なレポートは[こちら](https://wandb.ai/ayush-thakur/huggingface/reports/How-to-Fine-Tune-HuggingFace-Transformers-on-a-Custom-Dataset--Vmlldzo0MzQ2MDc)。
</details>

## 問題、質問、フィーチャーリクエスト

