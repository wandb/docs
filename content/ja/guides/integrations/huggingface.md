---
title: Hugging Face Transformers
menu:
  default:
    identifier: ja-guides-integrations-huggingface
    parent: integrations
weight: 110
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Optimize_Hugging_Face_models_with_Weights_&_Biases.ipynb" >}}

[Hugging Face Transformers](https://huggingface.co/transformers/) ライブラリを使うと、BERT のような最先端の NLP モデルや、mixed precision、gradient checkpointing のようなトレーニング手法を簡単に扱えます。[W&B integration](https://huggingface.co/transformers/main_classes/callback.html#transformers.integrations.WandbCallback) を使えば、その使いやすさを損なうことなく、リッチで柔軟な 実験管理 と モデルのバージョン管理 を、インタラクティブで中央集約型のダッシュボードに追加できます。

## 数行でリッチなロギング

```python
os.environ["WANDB_PROJECT"] = "<my-amazing-project>"  # W&B の Project 名を設定
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # すべてのモデルのチェックポイントをログする

from transformers import TrainingArguments, Trainer

args = TrainingArguments(..., report_to="wandb")  # W&B へのロギングを有効化
trainer = Trainer(..., args=args)
```
{{< img src="/images/integrations/huggingface_gif.gif" alt="Hugging Face ダッシュボード" >}}

{{% alert %}}
すぐに動くコードから始めたい場合は、この [Google Colab](https://wandb.me/hf) をチェックしてください。
{{% /alert %}}

## はじめに: 実験をトラッキング

### サインアップして API キー を作成

API キー は、あなたのマシンを W&B に対して認証します。API キー はユーザープロファイルから生成できます。

{{% alert %}}
よりスムーズに進めるには、[W&B authorization page](https://wandb.ai/authorize) に直接アクセスして API キー を生成してください。表示された API キー をコピーし、パスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上のユーザープロファイルアイコンをクリックします。
1. **User Settings** を選び、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックします。表示された API キー をコピーしてください。API キー を隠すにはページを再読み込みします。

### `wandb` ライブラリをインストールしてログイン

ローカルに `wandb` ライブラリをインストールしてログインするには:

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) にあなたの API キー を設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb` ライブラリをインストールしてログインします。



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

W&B を初めて使う場合は、[クイックスタート]({{< relref path="/guides/quickstart.md" lang="ja" >}}) もご覧ください。


### プロジェクトに名前を付ける

W&B の Project には、関連する Runs からログされたすべてのチャート、データ、モデルが保存されます。Project に名前を付けると、作業を整理でき、1 つの Project に関するすべての情報を 1 か所にまとめられます。

Run を Project に追加するには、`WANDB_PROJECT` 環境変数に Project 名を設定するだけです。`WandbCallback` はこの Project 名の環境変数を拾って、run のセットアップ時に使います。

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

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
`Trainer` を初期化する _前に_ Project 名を設定してください。
{{% /alert %}}

Project 名を指定しない場合、デフォルトの Project 名は `huggingface` です。

### トレーニングの Runs を W&B にログする

`Trainer` のトレーニング引数を指定するときの、コード内でもコマンドラインでも、**最重要ステップ** は `report_to` を `"wandb"` に設定して W&B へのロギングを有効化することです。

`TrainingArguments` の `logging_steps` 引数で、トレーニング中にどれくらいの頻度で W&B にメトリクスを送るかを制御できます。`run_name` 引数を使えば、W&B 上のトレーニング run に名前を付けることもできます。

以上です。これで、トレーニング中に損失、評価メトリクス、モデルのトポロジー、勾配が W&B にログされます。

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

```bash
python run_glue.py \     # あなたの Python スクリプトを実行
  --report_to wandb \    # W&B へのロギングを有効化
  --run_name bert-base-high-lr \   # W&B の run 名（任意）
  # 他のコマンドライン引数
```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```python
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    # その他の引数やキーワード引数
    report_to="wandb",  # W&B へのロギングを有効化
    run_name="bert-base-high-lr",  # W&B の run 名（任意）
    logging_steps=1,  # W&B へのログ頻度
)

trainer = Trainer(
    # その他の引数やキーワード引数
    args=args,  # トレーニング引数
)

trainer.train()  # トレーニング開始（W&B にロギング）
```

{{% /tab %}}
{{< /tabpane >}}

{{% alert %}}
TensorFlow を使っていますか？PyTorch の `Trainer` を TensorFlow の `TFTrainer` に置き換えるだけです。
{{% /alert %}}

### モデルのチェックポイントを有効化する


[Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) を使うと、Models と Datasets を最大 100GB まで無料で保存でき、その後 W&B の [Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) を利用できます。Registry を使うと、Models を登録して探索・評価したり、ステージングの準備をしたり、プロダクション環境にデプロイできます。

Hugging Face のモデルのチェックポイントを Artifacts にログするには、`WANDB_LOG_MODEL` 環境変数を次の _いずれか一つ_ に設定します:

- **`checkpoint`**: [`TrainingArguments`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments) の `args.save_steps` ごとにチェックポイントをアップロードします。 
- **`end`**: `load_best_model_at_end` も設定されている場合、トレーニングの最後にモデルをアップロードします。
- **`false`**: モデルをアップロードしません。


{{< tabpane text=true >}}

{{% tab header="コマンドライン" value="cli" %}}

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

以降に初期化する任意の Transformers の `Trainer` は、Models をあなたの W&B Project にアップロードします。ログしたモデルのチェックポイントは [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) の UI から閲覧でき、完全なモデルのリネージを含みます（UI でのモデルチェックポイントの例は [こちら](https://wandb.ai/wandb/arttest/artifacts/model/iv3_trained/5334ab69740f9dda4fed/lineage?_gl=1*yyql5q*_ga*MTQxOTYyNzExOS4xNjg0NDYyNzk1*_ga_JH1SJHJQXJ*MTY5MjMwNzI2Mi4yNjkuMS4xNjkyMzA5NjM2LjM3LjAuMA..)）。

{{% alert %}}
デフォルトでは、`WANDB_LOG_MODEL` が `end` のときは `model-{run_id}`、`checkpoint` のときは `checkpoint-{run_id}` という名前で W&B Artifacts に保存されます。
ただし、`TrainingArguments` に [`run_name`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.run_name) を渡した場合は、`model-{run_name}` または `checkpoint-{run_name}` という名前で保存されます。
{{% /alert %}}

#### W&B Registry
チェックポイントを Artifacts にログしたら、[Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) にベストなモデルチェックポイントを登録して、チーム全体で一元管理できます。Registry では、タスクごとにベストな Models を整理し、モデルのライフサイクルを管理し、ML のライフサイクル全体を追跡・監査し、下流のアクションを[自動化]({{< relref path="/guides/core/automations/" lang="ja" >}})できます。

モデルの Artifact をリンクする方法は、[Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) を参照してください。
 
### トレーニング中の評価出力を可視化する

トレーニングや評価中にモデルの出力を可視化することは、モデルの学習状況を深く理解するうえで欠かせません。

Transformers の Trainer で callbacks システムを使うことで、モデルのテキスト生成出力やその他の予測などの有用なデータを、W&B に、あるいは W&B Tables にログできます。 

トレーニングしながら評価出力を W&B Table にログする方法の完全なガイドは、以下の [Custom logging のセクション]({{< relref path="#custom-logging-log-and-view-evaluation-samples-during-training" lang="ja" >}}) を参照してください:


{{< img src="/images/integrations/huggingface_eval_tables.png" alt="評価出力を表示する W&B Table の例" >}}

### W&B の Run を終了する（Notebooks のみ） 

トレーニングが Python スクリプトにまとめられている場合、スクリプトが終了すると W&B の run も終了します。

Jupyter や Google Colab のノートブックを使う場合は、`run.finish()` を呼び出してトレーニングの終了を知らせてください。

```python
run = wandb.init()
trainer.train()  # トレーニング開始（W&B にロギング）

# トレーニング後の分析、テスト、その他のロギングコード

run.finish()
```

### 結果を可視化する

トレーニング結果をログしたら、[W&B Dashboard]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) で動的に探索できます。多数の Runs を一度に比較したり、興味深い学びにズームインしたり、柔軟でインタラクティブな可視化で複雑なデータからインサイトを引き出せます。

## 高度な機能と FAQ

### ベストモデルはどう保存すればいいですか？
`load_best_model_at_end=True` を指定した `TrainingArguments` を `Trainer` に渡すと、W&B はベストなモデルのチェックポイントを Artifacts に保存します。

モデルのチェックポイントを Artifacts に保存していれば、それらを [Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) に昇格できます。Registry では次のことができます:
- ML タスクごとにベストなモデルバージョンを整理する。
- モデルを一元化してチームと共有する。
- プロダクションに向けてステージングしたり、追加評価のためにブックマークする。
- 下流の CI/CD プロセスをトリガーする。

### 保存済みモデルはどう読み込めばいいですか？

`WANDB_LOG_MODEL` でモデルを W&B Artifacts に保存していれば、追加のトレーニングや推論のためにモデルの重みをダウンロードできます。以前と同じ Hugging Face のアーキテクチャーに読み戻してください。

```python
# 新しい run を作成
with wandb.init(project="amazon_sentiment_analysis") as run:
    # Artifact の名前とバージョンを渡す
    my_model_name = "model-bert-base-high-lr:latest"
    my_model_artifact = run.use_artifact(my_model_name)

    # モデル重みをフォルダにダウンロードし、パスを返す
    model_dir = my_model_artifact.download()

    # そのフォルダから Hugging Face モデルを読み込む
    #  同じモデルクラスを使う
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=num_labels
    )

    # 追加トレーニング、または推論を実行
```

### チェックポイントからトレーニングを再開するには？
`WANDB_LOG_MODEL='checkpoint'` を設定していた場合、`TrainingArguments` の `model_name_or_path` 引数に `model_dir` を使い、`Trainer` に `resume_from_checkpoint=True` を渡すことでトレーニングを再開できます。

```python
last_run_id = "xxxxxxxx"  # wandb Workspace から run_id を取得

# run_id から W&B の run を再開
with wandb.init(
    project=os.environ["WANDB_PROJECT"],
    id=last_run_id,
    resume="must",
) as run:
    # Artifact を run に接続
    my_checkpoint_name = f"checkpoint-{last_run_id}:latest"
    my_checkpoint_artifact = run.use_artifact(my_model_name)

    # チェックポイントをフォルダにダウンロードし、パスを返す
    checkpoint_dir = my_checkpoint_artifact.download()

    # モデルと trainer を再初期化
    model = AutoModelForSequenceClassification.from_pretrained(
        "<model_name>", num_labels=num_labels
    )
    # ここに素晴らしいトレーニング引数を設定
    training_args = TrainingArguments()

    trainer = Trainer(model=model, args=training_args)

    # チェックポイントから再開するために checkpoint_dir を使う
    trainer.train(resume_from_checkpoint=checkpoint_dir)
```

### トレーニング中に評価サンプルをログして可視化するには

Transformers の `Trainer` から W&B へのロギングは、Transformers ライブラリの [`WandbCallback`](https://huggingface.co/transformers/main_classes/callback.html#transformers.integrations.WandbCallback) が担当します。Hugging Face 側のロギングをカスタマイズしたい場合は、`WandbCallback` をサブクラス化し、Trainer クラスの追加メソッドを活用する機能を足してください。 

以下は、この新しい callback を HF の Trainer に追加する一般的なパターンで、その後に評価出力を W&B Table にログするコード完全版の例を示します:


```python
# いつも通り Trainer を初期化
trainer = Trainer()

# Trainer オブジェクトを渡して新しいロギング callback を初期化
evals_callback = WandbEvalsCallback(trainer, tokenizer, ...)

# Trainer に callback を追加
trainer.add_callback(evals_callback)

# いつも通り Trainer のトレーニングを開始
trainer.train()
```

#### トレーニング中の評価サンプルを表示する

このセクションでは、`WandbCallback` をカスタマイズして、トレーニング中にモデルの予測を実行し、評価サンプルを W&B Table にログする方法を紹介します。`on_evaluate` メソッドを使い、毎回の `eval_steps` で実行します。

ここでは、トークナイザーを使ってモデル出力から予測とラベルをデコードする `decode_predictions` 関数を作成しています。

続いて、予測とラベルから pandas の DataFrame を作成し、`epoch` 列を追加します。

最後に、DataFrame から `wandb.Table` を作り、wandb にログします。
さらに、`freq` エポックごとに予測をログすることで、ロギング頻度を制御できます。

**注意**: 通常の `WandbCallback` と異なり、このカスタム callback は `Trainer` の初期化時ではなく、`Trainer` のインスタンス化の **後に** trainer に追加する必要があります。これは、`Trainer` のインスタンスが初期化時に callback に渡されるためです。

```python
from transformers.integrations import WandbCallback
import pandas as pd


def decode_predictions(tokenizer, predictions):
    labels = tokenizer.batch_decode(predictions.label_ids)
    logits = predictions.predictions.argmax(axis=-1)
    prediction_text = tokenizer.batch_decode(logits)
    return {"labels": labels, "predictions": prediction_text}


class WandbPredictionProgressCallback(WandbCallback):
    """トレーニング中のモデル予測をログするためのカスタム WandbCallback。

    このコールバックは、トレーニングの各ロギングステップで
    モデルの予測とラベルを wandb.Table にログします。
    トレーニングの進行に伴うモデル予測の可視化を可能にします。

    Attributes:
        trainer (Trainer): Hugging Face の Trainer インスタンス。
        tokenizer (AutoTokenizer): モデルに対応するトークナイザー。
        sample_dataset (Dataset): 予測生成用に検証データセットから抽出したサブセット。
        num_samples (int, optional): 予測生成に使う検証データのサンプル数。デフォルトは 100。
        freq (int, optional): ロギング頻度（エポック単位）。デフォルトは 2。
    """

    def __init__(self, trainer, tokenizer, val_dataset, num_samples=100, freq=2):
        """WandbPredictionProgressCallback インスタンスを初期化。

        Args:
            trainer (Trainer): Hugging Face の Trainer インスタンス。
            tokenizer (AutoTokenizer): モデルに対応するトークナイザー。
            val_dataset (Dataset): 検証データセット。
            num_samples (int, optional): 予測生成に使う検証データのサンプル数。
              デフォルトは 100。
            freq (int, optional): ロギング頻度。デフォルトは 2。
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset.select(range(num_samples))
        self.freq = freq

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        # ロギング頻度を制御（予測を `freq` エポックごとにログする）
        if state.epoch % self.freq == 0:
            # 予測を生成
            predictions = self.trainer.predict(self.sample_dataset)
            # 予測とラベルをデコード
            predictions = decode_predictions(self.tokenizer, predictions)
            # 予測を wandb.Table に追加
            predictions_df = pd.DataFrame(predictions)
            predictions_df["epoch"] = state.epoch
            records_table = self._wandb.Table(dataframe=predictions_df)
            # テーブルを wandb にログ
            self._wandb.log({"sample_predictions": records_table})


# まず Trainer を初期化
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
)

# WandbPredictionProgressCallback を初期化
progress_callback = WandbPredictionProgressCallback(
    trainer=trainer,
    tokenizer=tokenizer,
    val_dataset=lm_dataset["validation"],
    num_samples=10,
    freq=2,
)

# trainer に callback を追加
trainer.add_callback(progress_callback)
```

より詳細な例はこの [colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Custom_Progress_Callback.ipynb) を参照してください。


### 利用できる追加の W&B 設定は？

`Trainer` で何をログするかは、環境変数でさらに細かく設定できます。W&B の環境変数一覧は[こちら]({{< relref path="/guides/hosting/env-vars.md" lang="ja" >}})にあります。

| Environment Variable | Usage |
| -------------------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `WANDB_PROJECT`      | Project に名前を付けます（デフォルトは `huggingface`）                                                                                                                                                                                                                                                      |
| `WANDB_LOG_MODEL`    | <p>モデルのチェックポイントを W&B Artifact としてログします（デフォルトは `false`）</p><ul><li><code>false</code>（デフォルト）: モデルのチェックポイントを保存しません </li><li><code>checkpoint</code>: Trainer の TrainingArguments で設定した args.save_steps ごとにチェックポイントをアップロードします。</li><li><code>end</code>: トレーニング終了時に最終チェックポイントをアップロードします。</li></ul> |
| `WANDB_WATCH`        | <p>モデルの勾配やパラメータをログするかどうかを設定します</p><ul><li><code>false</code>（デフォルト）: 勾配やパラメータをログしません </li><li><code>gradients</code>: 勾配のヒストグラムをログします </li><li><code>all</code>: 勾配とパラメータのヒストグラムをログします</li></ul> |
| `WANDB_DISABLED`     | `true` に設定するとロギングを完全にオフにします（デフォルトは `false`） |
| `WANDB_QUIET`.       | `true` に設定すると標準出力へのメッセージを重要なもののみに制限します（デフォルトは `false`）                                                                                                                                                                                                                                     |
| `WANDB_SILENT`       | `true` に設定すると wandb の出力をサイレントにします（デフォルトは `false`）                                                                                                                                                                                                                                |

{{< tabpane text=true >}}

{{% tab header="コマンドライン" value="cli" %}}

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


### `wandb.init` をカスタマイズするには？

`Trainer` が使う `WandbCallback` は、`Trainer` の初期化時に内部で `wandb.init` を呼び出します。代わりに、`Trainer` を初期化する前に手動で `wandb.init` を呼び出して Runs をセットアップすることもできます。これにより、W&B の run 設定を完全に制御できます。

`init` に渡せる例を以下に示します。`wandb.init()` の詳細は、[`wandb.init()` リファレンス]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) を参照してください。

```python
wandb.init(
    project="amazon_sentiment_analysis",
    name="bert-base-high-lr",
    tags=["baseline", "high-lr"],
    group="bert",
)
```


## 追加リソース

Transformers と W&B に関連する記事を 6 つご紹介します

<details>

<summary>Hugging Face Transformers のハイパーパラメータ最適化</summary>

* Hugging Face Transformers のハイパーパラメータ最適化について、Grid Search、Bayesian Optimization、Population Based Training の 3 手法を比較します。
* Hugging Face transformers の標準の uncased BERT モデルを使い、SuperGLUE ベンチマークの RTE データセットでファインチューニングします。
* 結果として、Population Based Training が本稿の Hugging Face transformer モデルのハイパーパラメータ最適化に最も有効であることが示されました。

[Hyperparameter Optimization for Hugging Face Transformers report](https://wandb.ai/amogkam/transformers/reports/Hyperparameter-Optimization-for-Hugging-Face-Transformers--VmlldzoyMTc2ODI) を読む
</details>

<details>

<summary>Hugging Tweets: ツイートを生成するモデルをトレーニング</summary>

* 本記事では、学習済みの GPT2 HuggingFace Transformer モデルを、任意のユーザーのツイートで 5 分でファインチューニングする方法を紹介します。
* モデルのパイプラインは、ツイートのダウンロード、データセットの最適化、初期実験、ユーザー間の損失比較、モデルのファインチューニング、という流れです。

全文は [こちらのレポート](https://wandb.ai/wandb/huggingtweets/reports/HuggingTweets-Train-a-Model-to-Generate-Tweets--VmlldzoxMTY5MjI) を参照
</details>

<details>

<summary>Sentence Classification With Hugging Face BERT and W&B</summary>

* 近年の自然言語処理のブレークスルーを活かし、転移学習の NLP への応用に焦点を当てた文分類器を構築します。
* 単文分類用に The Corpus of Linguistic Acceptability (CoLA) データセットを使用します。これは、文法的に正しいかどうかのラベルが付いた文の集合で、2018 年 5 月に初公開されました。
* Google の BERT を使って、幅広い NLP タスクで最小限の労力で高性能なモデルを作成します。

全文は [こちら](https://wandb.ai/cayush/bert-finetuning/reports/Sentence-Classification-With-Huggingface-BERT-and-W-B--Vmlldzo4MDMwNA) を参照
</details>

<details>

<summary>Hugging Face モデルのパフォーマンスを追跡するステップバイステップガイド</summary>

* W&B と Hugging Face transformers を使って DistilBERT を GLUE ベンチマークでトレーニングします。DistilBERT は BERT より 40% 小さく、BERT の精度の 97% を保持します。
* GLUE ベンチマークは、NLP モデルのトレーニング用に 9 つのデータセットとタスクを集めたものです。

全文は [こちら](https://wandb.ai/jxmorris12/huggingface-demo/reports/A-Step-by-Step-Guide-to-Tracking-HuggingFace-Model-Performance--VmlldzoxMDE2MTU) を参照
</details>

<details>

<summary>HuggingFace における Early Stopping の例</summary>

* Early Stopping 正則化を使った Hugging Face Transformer のファインチューニングは、PyTorch と TensorFlow の両方でネイティブに実行できます。
* TensorFlow では `tf.keras.callbacks.EarlyStopping` コールバックを使えば簡単です。
* PyTorch にはすぐに使える Early Stopping メソッドはありませんが、GitHub Gist に動作する Early Stopping フックがあります。

全文は [こちら](https://wandb.ai/ayush-thakur/huggingface/reports/Early-Stopping-in-HuggingFace-Examples--Vmlldzo0MzE2MTM) を参照
</details>

<details>

<summary>カスタム Dataset で Hugging Face Transformers をファインチューニングする方法</summary>

カスタム IMDB Dataset に対するセンチメント分析（二値分類）のために DistilBERT transformer をファインチューニングします。

全文は [こちら](https://wandb.ai/ayush-thakur/huggingface/reports/How-to-Fine-Tune-HuggingFace-Transformers-on-a-Custom-Dataset--Vmlldzo0MzQ2MDc) を参照
</details>

## ヘルプや機能リクエスト

Hugging Face と W&B のインテグレーションに関する問題、質問、または機能リクエストは、[Hugging Face forums のこちらのスレッド](https://discuss.huggingface.co/t/logging-experiment-tracking-with-w-b/498)に投稿するか、Hugging Face の [Transformers GitHub repo](https://github.com/huggingface/transformers) で issue を作成してください。