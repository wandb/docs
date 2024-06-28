---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';


# Hugging Face Transformers

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Optimize_Hugging_Face_models_with_Weights_&_Biases.ipynb"></CTAButtons>

[Hugging Face Transformers](https://huggingface.co/transformers/) ライブラリは、BERTのような最新のNLPモデルや混合精度と勾配チェックポイントのようなトレーニング技術を簡単に利用できるようにします。[W&Bインテグレーション](https://huggingface.co/transformers/main_classes/callback.html#transformers.integrations.WandbCallback)により、使いやすさを損なうことなく、リッチで柔軟な実験管理とモデルバージョン管理をインタラクティブな中央ダッシュボードに追加します。

## 🤗 数行で次レベルのログ記録

```python
os.environ["WANDB_PROJECT"] = "<my-amazing-project>"  # W&Bプロジェクトの名前を設定
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # すべてのモデルチェックポイントをログ

from transformers import TrainingArguments, Trainer

args = TrainingArguments(..., report_to="wandb")  # W&Bログを有効にする
trainer = Trainer(..., args=args)
```
![実験の結果をW&Bのインタラクティブダッシュボードで探索](@site/static/images/integrations/huggingface_gif.gif)

:::info
すぐに動作するコードに飛びたい場合は、こちらの[Google Colab](https://wandb.me/hf)を確認してください。
:::

## 始めましょう: 実験を追跡

### 1) サインアップ、`wandb`ライブラリのインストール、およびログイン

a) 無料アカウントに[**サインアップ**](https://wandb.ai/site)

b) `wandb`ライブラリをpipでインストール

c) トレーニングスクリプトでログインするには、www.wandb.aiでアカウントにサインインしている必要があります。その後、[**Authorizeページ**](https://wandb.ai/authorize)でAPIキーを見つけることができます。

Weights and Biasesを初めて使用する場合は、[**クイックスタート**](../../quickstart.md)を確認してください。

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

### 2) プロジェクトの名前を設定

[プロジェクト](../app/pages/project-page.md)は、関連するrunsから記録されたすべてのチャート、データ、モデルが保存される場所です。プロジェクトに名前を付けることで、作業を整理し、1つのプロジェクトに関するすべての情報を1か所にまとめることができます。

プロジェクトにrunを追加するには、`WANDB_PROJECT`環境変数をプロジェクトの名前に設定するだけです。`WandbCallback`はこのプロジェクト名の環境変数をピックアップし、runの設定時に使用します。

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

プロジェクト名が指定されていない場合、プロジェクト名はデフォルトで "huggingface" になります。

### 3) トレーニングrunsをW&Bにログ

これは**最も重要なステップ**です: あなたの`Trainer`トレーニング引数を定義する際に、コード内またはコマンドラインから `report_to` を `"wandb"` に設定することで、Weights & Biasesによるログを有効にします。

`TrainingArguments`の `logging_steps` 引数は、トレーニング中にどのくらいの頻度でトレーニングメトリクスがW&Bに送信されるかを制御します。また、W&Bでのトレーニングrunに名前を付けるために `run_name` 引数を使用することもできます。

これで完了です！モデルは、トレーニング中に、損失、評価メトリクス、モデルのトポロジー、勾配をWeights & Biasesにログします。

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python', value: 'python'},
    {label: 'Command Line', value: 'cli'},
  ]}>
  <TabItem value="cli">

```bash
python run_glue.py \     # Pythonスクリプトを実行
  --report_to wandb \    # W&Bへのログを有効にする
  --run_name bert-base-high-lr \   # W&B runの名前（オプション）
  # 他のコマンドライン引数はここに
```

  </TabItem>
  <TabItem value="python">

```python
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    # 他の引数やキーワード引数はここに
    report_to="wandb",  # W&Bへのログを有効にする
    run_name="bert-base-high-lr",  # W&B runの名前（オプション）
    logging_steps=1,  # W&Bにログする頻度
)

trainer = Trainer(
    # 他の引数やキーワード引数はここに
    args=args,  # トレーニング引数
)

trainer.train()  # トレーニングを開始し、W&Bにログ
```

  </TabItem>
</Tabs>

:::info
TensorFlowを使用していますか？PyTorchの`Trainer`をTensorFlowの`TFTrainer`に置き換えるだけです。
:::

### 4) モデルのチェックポイントを有効にする

Weights & Biases の[Artifacts](../artifacts)を使用すると、最大100GBのモデルやデータセットを無料で保存でき、Weights & Biases [Model Registry](../model_registry)を使ってプロダクション環境でのステージングやデプロイの準備のためにモデルを登録できます。

Hugging Face モデルのチェックポイントを Artifacts にログするには、`WANDB_LOG_MODEL` 環境変数を `end`、`checkpoint`、または `false` のいずれかに設定します:

-  **`checkpoint`**: [`TrainingArguments`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments) の `args.save_steps` ごとにチェックポイントがアップロードされます。
- **`end`**:  トレーニングの終了時にモデルがアップロードされます。

トレーニングの終了時にベストモデルをアップロードするには、`WANDB_LOG_MODEL`を`load_best_model_at_end`と一緒に使用します。


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

今後初期化されるすべてのTransformers `Trainer`は、モデルをW&Bプロジェクトにアップロードします。ログしたモデルチェックポイントは[Artifacts](../artifacts) UIを通じて表示可能で、完全なモデルリネージを含みます（UIでモデルチェックポイントの例は[こちら](https://wandb.ai/wandb/arttest/artifacts/model/iv3_trained/5334ab69740f9dda4fed/lineage?_gl=1*yyql5q*_ga*MTQxOTYyNzExOS4xNjg0NDYyNzk1*_ga_JH1SJHJQXJ*MTY5MjMwNzI2Mi4yNjkuMS4xNjkyMzA5NjM2LjM3LjAuMA..)）。

:::info
デフォルトでは、モデルが `WANDB_LOG_MODEL` が `end` に設定されている場合は `model-{run_id}` として、`WANDB_LOG_MODEL` が `checkpoint` に設定されている場合は `checkpoint-{run_id}` として W&B Artifacts に保存されます。
ただし、 `TrainingArguments` に `run_name` を渡した場合、モデルは `model-{run_name}` または `checkpoint-{run_name}` として保存されます。
:::

#### W&B Model Registry
Artifactsにチェックポイントをログしたら、Weights & Biases の**[Model Registry](../model_registry)**を使用して最良のモデルチェックポイントを登録し、チーム全体でそれらを集中管理することができます。ここでは、タスクごとに最良のモデルを整理し、モデルライフサイクルを管理し、MLのライフサイクル全体を簡単に追跡および監査し、Webhookやジョブを使って[自動化](https://docs.wandb.ai/guides/models/automation)することが可能です。

モデルArtifactをModel Registryにリンクする方法については、[Model Registry](../model_registry)のドキュメントを参照してください。

### 5) トレーニング中の評価出力を可視化

トレーニングや評価中にモデル出力を可視化することは、モデルがどのようにトレーニングされているかを理解する上で非常に重要です。

Transformers Trainerのコールバックシステムを使用して、モデルのテキスト生成出力やその他の予測をW&B Tablesにログするなど、W&Bに追加のデータをログできます。

以下の**[カスタムログセクション](#custom-logging-log-and-view-evaluation-samples-during-training)**で、トレーニング中に評価出力をW&B Tableにログする方法についてのガイドを確認してください:


![評価出力を含むW&B Tableを表示](/images/integrations/huggingface_eval_tables.png)

### 6) W&BのRunを終了する（ノートブックのみ）

トレーニングがPythonスクリプトでカプセル化されている場合、スクリプトが終了したときにW&Bのrunも終了します。

JupyterやGoogle Colabノートブックを使用している場合、トレーニングが終了したことを`wandb.finish()`を呼び出して知らせる必要があります。

```python
trainer.train()  # トレーニングを開始し、W&Bにログ

# トレーニング後の分析、テスト、その他のログコード

wandb.finish()
```

### 7) 結果を可視化

トレーニング結果をログしたら、[W&Bダッシュボード](../track/app.md)で動的に結果を探索できます。数十のrunsを一度に比較したり、興味深い学びにズームインしたり、柔軟でインタラクティブな可視化を使って複雑なデータから洞察を引き出すことができます。

## 高度な機能とよくある質問

### 最高のモデルを保存するにはどうすれば良いですか？

`Trainer`に渡された `TrainingArguments` 内で `load_best_model_at_end=True` が設定されている場合、W&Bは最高のパフォーマンスのモデルチェックポイントをArtifactsに保存します。

チーム全体で最高のモデルバージョンを集中管理し、MLタスクごとに整理し、プロダクションにステージングし、さらに評価するためにブックマークし、下流のModel CI/CDプロセスを開始する場合、モデルチェックポイントをArtifactsに保存することを確認してください。Artifactsにログされた後、これらのチェックポイントは[Model Registry](../model_registry/intro.md)に昇格させることができます。

### 保存したモデルを読み込む

`WANDB_LOG_MODEL`でモデルをW&B Artifactsに保存した場合、追加のトレーニングや推論を行うためにモデルの重みをダウンロードできます。以前使用した同じHugging Faceアーキテクチャにそれらを読み込むだけです。

```python
# 新しいrunを作成
with wandb.init(project="amazon_sentiment_analysis") as run:
    # Artifactの名前とバージョンを渡す
    my_model_name = "model-bert-base-high-lr:latest"
    my_model_artifact = run.use_artifact(my_model_name)

    # モデル重みをフォルダにダウンロードし、パスを返す
    model_dir = my_model_artifact.download()

    # 同じモデルクラスを使用して、そのフォルダからHugging Faceモデルをロード
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=num_labels
    )

    # 追加のトレーニングや推論を実行
```

### チェックポイントからトレーニングを再開
`WANDB_LOG_MODEL='checkpoint'`を設定している場合、`model_dir`を`TrainingArguments`の`model_name_or_path`引数として使用し、`Trainer`に`resume_from_checkpoint=True`を渡してトレーニングを再開することができます。

```python
last_run_id = "xxxxxxxx"  # wandbワークスペースからrun_idを取得

# run_idからwandbのrunを再開
with wandb.init(
    project=os.environ["WANDB_PROJECT"],
    id=last_run_id,
    resume="must",
) as run:
    # Artifactをrunに接続
    my_checkpoint_name = f"checkpoint-{last_run_id}:latest"
    my_checkpoint_artifact = run.use_artifact(my_model_name)

    # フォルダにチェックポイントをダウンロードし、パスを返す
    checkpoint_dir = my_checkpoint_artifact.download()

    # モデルとトレーナーを再初期化
    model = AutoModelForSequenceClassification.from_pretrained(
        "<model_name>", num_labels=num_labels
    )
    # ここにトレーニングの素晴らしい引数を設定'
    training_args = TrainingArguments()

    trainer = Trainer(model=model, args=training_args)

    # チェックポイントディレクトリを使用してトレーニングをチェックポイントから再開
    trainer.train(resume_from_checkpoint=checkpoint_dir)
```

### カスタムログ: トレーニング中に評価サンプルをログおよび表示

Transformersライブラリの[`WandbCallback`](https://huggingface.co/transformers/main_classes/callback.html#transformers.integrations.WandbCallback)を介してWeights & Biasesにログを行います。Hugging Faceのログをカスタマイズする必要がある場合は、`WandbCallback`をサブクラス化し、Trainerクラスからの追加メソッドを活用する追加機能を追加することで、このコールバックを変更できます。

以下に、新しいコールバックをHF Trainerに追加するための一般的なパターンを示し、その下にW&B Tableに評価出力をログするコード完結の例を示します:


```python
# 通常通りにTrainerをインスタンス化
trainer = Trainer()

# トレーナーオブジェクトを渡して新しいログコールバックをインスタンス化
evals_callback = WandbEvalsCallback(trainer, tokenizer, ...)

# トレーナーにコールバックを追加
trainer.add_callback(evals_callback)

# 通常通りにTrainerのトレーニングを開始
trainer.train()
```

#### トレーニング中の評価サンプルを表示

次のセクションでは、`WandbCallback` をカスタマイズして、トレーニング中にモデルの予測を実行し、評価サンプルを W&B Table にログする方法を示します。 トレーナーコールバックの `on_evaluate` メソッドを使用して、すべての `eval_steps` を実行します。

ここでは、トークナイザーを使ってモデル出力から予測とラベルをデコードする `decode_predictions` 関数を書きました。

次に、予測とラベルから pandas DataFrame を作成し、DataFrame に `epoch` カラムを追加します。

最後に、DataFrame から `wandb.Table` を作成し、それを wandb にログします。
さらに、予測をすべての `freq` エポックでログすることで、ログの頻度をコントロールできます。

**注**: 通常の `WandbCallback` と異なり、このカスタムコールバックは `Trainer` のインスタンス化後にトレーナーに追加する必要があります。これは、コールバックの初期化中に `Trainer` インスタンスがコールバックに渡されるためです。

```python
from transformers.integrations import WandbCallback
import pandas as pd


def decode_predictions(tokenizer, predictions):
    labels = tokenizer.batch_decode(predictions.label_ids)
    logits = predictions.predictions.argmax(axis=-1)
    prediction_text = tokenizer.batch_decode(logits)
    return {"labels": labels, "predictions": prediction_text}


class WandbPredictionProgressCallback(WandbCallback):
    """トレーニング中のモデル予測をログするカスタムWandbCallback

    このコールバックは、トレーニング中の各ログステップでモデルの予測とラベルを wandb.Table にログします。
    トレーニングの進行に応じてモデルの予測を視覚化できます。

    属性:
        trainer (Trainer): Hugging Face Trainer インスタンス.
        tokenizer (AutoTokenizer): モデルに関連付けられたトークナイザー.
        sample_dataset (Dataset): 予測生成のための検証データセットのサブセット.
        num_samples (int, optional): 予測生成のために検証データセットから選択するサンプル数. デフォルトは100.
        freq (int, optional): ログの頻度. デフォルトは2.
    """

    def __init__(self, trainer, tokenizer, val_dataset,
                 num_samples=100, freq=2):
        """WandbPredictionProgressCallback インスタンスを初期化

        引数:
            trainer (Trainer): Hugging Face Trainer インスタンス.
            tokenizer (AutoTokenizer): モデルに関連付けられたトークナイザー.
            val_dataset (Dataset): 検証データセット.
            num_samples (int, optional): 予測生成のために検証データセットから選択するサンプル数. 
              デフォルトは100.
            freq (int, optional): ログの頻度. デフォルトは2.
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset.select(range(num_samples))
        self.freq = freq

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        # 予測をすべての `freq` エポックでログすることでログの頻度をコントロール
        if state.epoch % self.freq == 0:
            # 予測生成
            predictions = self.trainer.predict(self.sample_dataset)
            # 予測とラベルのデコード
            predictions = decode_predictions(self.tokenizer, predictions)
            # 予測を wandb.Table に追加
            predictions_df = pd.DataFrame(predictions)
            predictions_df["epoch"] = state.epoch
            records_table = self._wandb.Table(dataframe=predictions_df)
            # テーブルを wandb にログ
            self._wandb.log({"sample_predictions": records_table})


# まず、Trainer をインスタンス化
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
)

# WandbPredictionProgressCallback のインスタンス化
progress_callback = WandbPredictionProgressCallback(
    trainer=trainer,
    tokenizer=tokenizer,
    val_dataset=lm_dataset["validation"],
    num_samples=10,
    freq=2,
)

# トレーナーにコールバックを追加
trainer.add_callback(progress_callback)
```

詳細な例については、この [colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Custom_Progress_Callback.ipynb) を参照してください。

### 追加の W&B 設定

環境変数を設定することで、`Trainer` にログされる内容をさらに設定できます。 W&B 環境変数の完全なリストは [こちら](https://docs.wandb.ai/library/environment-variables) で確認できます。

| 環境変数               | 使い方                                                                                                                                                                                                                                                                                                    |
| -------------------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `WANDB_PROJECT`      | プロジェクトに名前を付けます（デフォルトは `huggingface`）                                                                                                                                                                                                                                                      |
| `WANDB_LOG_MODEL`    | <p>モデルのチェックポイントを W&B Artifact としてログします（デフォルトは `false`）</p><ul><li><code>false</code>: モデルのチェックポイントはログしません。</li><li><code>checkpoint</code>: args.save_steps ごとにチェックポイントがアップロードされます（Trainer の TrainingArguments に設定されています）。</li><li><code>end</code>: トレーニングの終了時に最終モデルのチェックポイントがアップロードされます。</li></ul>                                                                                                                                                                                                                                   |
| `WANDB_WATCH`        | <p>モデルの勾配、パラメータ、またはその両方をログするかどうかを設定します。</p><ul><li><code>false</code> (デフォルト): 勾配またはパラメータをログしません。</li><li><code>gradients</code>: 勾配のヒストグラムをログします。</li><li><code>all</code>: 勾配とパラメータのヒストグラムをログします。</li></ul> |
| `WANDB_DISABLED`     | ログ全体を無効にするには `true` に設定します（デフォルトは `false`）                                                                                                                                                                                                                                           |
| `WANDB_SILENT`       | wandb によって出力されるメッセージを抑制するには `true` に設定します（デフォルトは `false`）                                                                                                                                                                                                                                |

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

### `wandb.init` のカスタマイズ

`Trainer` が初期化されるときに、`Trainer` が使用する `WandbCallback` が内部で `wandb.init` を呼び出します。代わりに、`Trainer` を初期化する前に `wandb.init` を手動で呼び出して、Runs をセットアップすることもできます。これにより、W&B Run の設定を完全にコントロールできます。

`init` に渡す例は以下の通りです。`wandb.init` の使用方法について詳しくは、[リファレンスドキュメント](../../ref/python/init.md) をご覧ください。

```python
wandb.init(
    project="amazon_sentiment_analysis",
    name="bert-base-high-lr",
    tags=["baseline", "high-lr"],
    group="bert",
)
```

## 注目の記事

以下は Transformers と W&B に関連する 6 本の記事です:

<details>

<summary>Hugging Face Transformers のハイパーパラメータ最適化</summary>

* Hugging Face Transformers のハイパーパラメータ最適化のための 3 つの戦略 - グリッド検索、ベイズ最適化、および Population Based Training を比較します。
* Hugging Face transformers から標準的な未ケースの BERT モデルを使用し、SuperGLUE ベンチマークから RTE データセットにファインチューンします。
* 結果は、Population Based Training が Hugging Face transformer モデルのハイパーパラメータ最適化に最も効果的であることを示しています。

完全なレポートを読む [こちら](https://wandb.ai/amogkam/transformers/reports/Hyperparameter-Optimization-for-Hugging-Face-Transformers--VmlldzoyMTc2ODI).
</details>

<details>

<summary>Hugging Tweets: ツイート生成モデルのトレーニング</summary>

* 記事では、著者が 5 分で誰のツイートでも生成できる事前学習済み GPT2 HuggingFace Transformer モデルをファインチューンする方法を説明しています。
* モデルは以下のパイプラインを利用します: ツイートのダウンロード、データセットの最適化、初期実験、ユーザー間の損失比較、モデルのファインチューン。

完全なレポートを読む [こちら](https://wandb.ai/wandb/huggingtweets/reports/HuggingTweets-Train-a-Model-to-Generate-Tweets--VmlldzoxMTY5MjI).
</details>

<details>

<summary>Sentence Classification With Hugging Face BERT and WB</summary>

* この記事では、最近の自然言語処理における画期的な技術を活用し、転移学習を NLP に応用することに焦点を当てて、文分類器を構築します。
* 文単位の分類のために、文が文法的に正しいか間違っているかをラベル付けした 2018 年 5 月に初めて公開された CoLA データセットを使用します。
* Google's BERT を使用して、さまざまな NLP タスクで最小限の労力で高性能なモデルを作成します。

完全なレポートを読む [こちら](https://wandb.ai/cayush/bert-finetuning/reports/Sentence-Classification-With-Huggingface-BERT-and-W-B--Vmlldzo4MDMwNA).
</details>

<details>

<summary>A Step by Step Guide to Tracking Hugging Face Model Performance</summary>

* Weights & Biases と Hugging Face transformers を使用して、BERT の 40% のサイズで 97% の精度を保つ Transformer である DistilBERT を GLUE ベンチマークでトレーニング
* GLUE ベンチマークは、NLP モデルをトレーニングするための 9 つのデータセットとタスクのコレクションです

完全なレポートを読む [こちら](https://wandb.ai/jxmorris12/huggingface-demo/reports/A-Step-by-Step-Guide-to-Tracking-HuggingFace-Model-Performance--VmlldzoxMDE2MTU).
</details>

<details>

<summary>Early Stopping in HuggingFace - Examples</summary>

* PyTorch または TensorFlow のネイティブな Early Stopping 正規化を使用して Hugging Face Transformer をファインチューンできます。
* TensorFlow では `tf.keras.callbacks.EarlyStopping` コールバックを使用すると簡単です。
* PyTorch には市販の early stopping メソッドはありませんが、GitHub Gist で使用できる early stopping フックがあります。

完全なレポートを読む [こちら](https://wandb.ai/ayush-thakur/huggingface/reports/Early-Stopping-in-HuggingFace-Examples--Vmlldzo0MzE2MTM).
</details>

<details>

<summary>How to Fine-Tune Hugging Face Transformers on a Custom Dataset</summary>

カスタム IMDB データセットでセントメント分析（二項分類）のために DistilBERT トランスフォーマーをファインチューンします。

完全なレポートを読む [こちら](https://wandb.ai/ayush-thakur/huggingface/reports/How-to-Fine-Tune-HuggingFace-Transformers-on-a-Custom-Dataset--Vmlldzo0MzQ6MDC).
</details>

## 問題、質問、機能リクエスト

Hugging Face W&B インテグレーションに関する問題、質問、または機能リクエストについては、この [Hugging Face フォーラムのスレッド](https://discuss.huggingface.co/t/logging-experiment-tracking-with-w-b/498) で投稿するか、Hugging Face [Transformers GitHub リポジトリ](https://github.com/huggingface/transformers) で issue をオープンしてください。