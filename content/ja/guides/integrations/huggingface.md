---
title: Hugging Face Transformers
menu:
  default:
    identifier: ja-guides-integrations-huggingface
    parent: integrations
weight: 110
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Optimize_Hugging_Face_models_with_Weights_&_Biases.ipynb" >}}

[Hugging Face Transformers](https://huggingface.co/transformers/) ライブラリを使用すると、最先端のNLPモデル（BERTなど）や、混合精度や勾配チェックポイントなどのトレーニング手法を簡単に使用できます。[W&B integration](https://huggingface.co/transformers/main_classes/callback.html#transformers.integrations.WandbCallback) により、使いやすさを損なうことなく、インタラクティブな集中型ダッシュボードに豊富で柔軟な実験管理とモデルの バージョン管理が追加されます。

## 数行でネクストレベルのロギング

```python
os.environ["WANDB_PROJECT"] = "<my-amazing-project>"  # W&B プロジェクトに名前を付ける
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # すべてのモデルチェックポイントをログに記録

from transformers import TrainingArguments, Trainer

args = TrainingArguments(..., report_to="wandb")  # W&B ロギングをオンにする
trainer = Trainer(..., args=args)
```
{{< img src="/images/integrations/huggingface_gif.gif" alt="W&B インタラクティブダッシュボードで実験結果を探索する" >}}

{{% alert %}}
すぐにコードに着手したい場合は、こちらの [Google Colab](https://wandb.me/hf) をご覧ください。
{{% /alert %}}

## はじめに: 実験を追跡する

### サインアップして API キーを作成する

API キーは、W&B に対してお客様のマシンを認証します。API キーは、ユーザープロファイルから生成できます。

{{% alert %}}
より効率的なアプローチとして、[https://wandb.ai/authorize](https://wandb.ai/authorize) に直接アクセスして API キーを生成できます。表示された API キーをコピーして、パスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上隅にあるユーザープロファイルアイコンをクリックします。
2. [**User Settings**] を選択し、[**API Keys**] セクションまでスクロールします。
3. [**Reveal**] をクリックします。表示された API キーをコピーします。API キーを非表示にするには、ページをリロードします。

### `wandb` ライブラリをインストールしてログインする

`wandb` ライブラリをローカルにインストールしてログインするには:

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) をお客様の API キーに設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

2. `wandb` ライブラリをインストールしてログインします。



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

W&B を初めて使用する場合は、[**クイックスタート**]({{< relref path="/guides/quickstart.md" lang="ja" >}}) を確認してください。


### プロジェクトに名前を付ける

W&B の Project は、関連する run からログに記録されたすべてのグラフ、データ、およびモデルが保存される場所です。プロジェクトに名前を付けると、作業を整理し、1 つのプロジェクトに関するすべての情報を 1 か所にまとめて管理できます。

run をプロジェクトに追加するには、`WANDB_PROJECT` 環境変数をプロジェクトの名前に設定するだけです。`WandbCallback` は、このプロジェクト名環境変数を取得し、run の設定時に使用します。

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
`Trainer` を初期化する _前に_ プロジェクト名を設定してください。
{{% /alert %}}

プロジェクト名が指定されていない場合、プロジェクト名はデフォルトで `huggingface` になります。

### トレーニング run を W&B に記録する

コード内またはコマンドラインから `Trainer` のトレーニング引数を定義する際に **最も重要なステップ** は、W&B でのロギングを有効にするために `report_to` を `"wandb"` に設定することです。

`TrainingArguments` の `logging_steps` 引数は、トレーニング中にトレーニング メトリクスが W&B にプッシュされる頻度を制御します。`run_name` 引数を使用して、W&B でトレーニング run に名前を付けることもできます。

以上です。これで、モデルはトレーニング中に損失、評価メトリクス、モデルトポロジ、および勾配を W&B にログ記録します。

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

```bash
python run_glue.py \     # Python スクリプトを実行する
  --report_to wandb \    # W&B へのロギングを有効にする
  --run_name bert-base-high-lr \   # W&B run の名前 (オプション)
  # その他のコマンドライン引数
```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```python
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    # その他の引数とキーワード引数
    report_to="wandb",  # W&B へのロギングを有効にする
    run_name="bert-base-high-lr",  # W&B run の名前 (オプション)
    logging_steps=1,  # W&B へのログ頻度
)

trainer = Trainer(
    # その他の引数とキーワード引数
    args=args,  # トレーニング引数
)

trainer.train()  # トレーニングを開始し、W&B にログを記録する
```

{{% /tab %}}
{{< /tabpane >}}

{{% alert %}}
TensorFlow を使用していますか？PyTorch の `Trainer` を TensorFlow の `TFTrainer` に交換するだけです。
{{% /alert %}}

### モデルチェックポイントをオンにする


W&B の [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) を使用すると、最大 100 GB のモデルとデータセットを無料で保存し、W&B [モデルレジストリ]({{< relref path="/guides/models/registry/model_registry/" lang="ja" >}}) を使用してモデルを登録し、本番環境でのステージングまたはデプロイメントの準備をすることができます。

Hugging Face モデルのチェックポイントを Artifacts にログ記録するには、`WANDB_LOG_MODEL` 環境変数を `end`、`checkpoint`、または `false` のいずれかに設定します。

- **`checkpoint`**: チェックポイントは、[`TrainingArguments`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments) から `args.save_steps` ごとにアップロードされます。
- **`end`**: モデルはトレーニングの最後にアップロードされます。

トレーニングの最後に最適なモデルをアップロードするには、`WANDB_LOG_MODEL` を `load_best_model_at_end` と共に使用します。


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

これから初期化する Transformers `Trainer` は、モデルを W&B プロジェクトにアップロードします。ログに記録したモデルのチェックポイントは、[Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) UI で表示でき、完全なモデルリネージが含まれています (UI のモデルチェックポイントの例については、[こちら](https://wandb.ai/wandb/arttest/artifacts/model/iv3_trained/5334ab69740f9dda4fed/lineage?_gl=1*yyql5q*_ga*MTQxOTYyNzExOS4xNjg0NDYyNzk1*_ga_JH1SJHJQXJ*MTY5MjMwNzI2Mi4yNjkuMS4xNjkyMzA5NjM2LjM3LjAuMA..) を参照してください)。


{{% alert %}}
デフォルトでは、`WANDB_LOG_MODEL` が `end` に設定されている場合は、モデルは `model-{run_id}` として W&B Artifacts に保存され、`WANDB_LOG_MODEL` が `checkpoint` に設定されている場合は、`checkpoint-{run_id}` として保存されます。
ただし、[`run_name`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.run_name) を `TrainingArguments` で渡すと、モデルは `model-{run_name}` または `checkpoint-{run_name}` として保存されます。
{{% /alert %}}

#### W&B モデルレジストリ
チェックポイントを Artifacts にログ記録したら、最適なモデルのチェックポイントを登録し、**[モデルレジストリ]({{< relref path="/guides/models/registry/model_registry/" lang="ja" >}})** を使用してチーム全体で一元化できます。ここでは、タスクごとに最適なモデルを整理し、モデルのライフサイクルを管理し、ML ライフサイクル全体で簡単な追跡と監査を容易にし、Webhooks またはジョブでダウンストリームアクションを [自動化]({{< relref path="/guides/models/automations/project-scoped-automations/#create-a-webhook-automation" lang="ja" >}}) できます。

モデル Artifact をモデルレジストリにリンクする方法については、[モデルレジストリ]({{< relref path="/guides/models/registry/model_registry/" lang="ja" >}}) のドキュメントを参照してください。
 
### トレーニング中に評価出力を可視化する

トレーニングまたは評価中にモデル出力を可視化することは、モデルのトレーニング方法を実際に理解するために不可欠な場合がよくあります。

Transformers Trainer のコールバックシステムを使用すると、モデルのテキスト生成出力やその他の予測などの追加の役立つデータを W&B Tables に W&B にログ記録できます。

W&B Table にログ記録するために、トレーニング中に評価出力をログ記録する方法に関する完全なガイドについては、以下の **[カスタムロギングセクション]({{< relref path="#custom-logging-log-and-view-evaluation-samples-during-training" lang="ja" >}})** を参照してください。


{{< img src="/images/integrations/huggingface_eval_tables.png" alt="評価出力を含む W&B Table を表示する" >}}

### W&B Run を完了する (ノートブックのみ)

トレーニングが Python スクリプトにカプセル化されている場合、W&B run はスクリプトが完了すると終了します。

Jupyter または Google Colab ノートブックを使用している場合は、`wandb.finish()` を呼び出して、トレーニングが完了したことを知らせる必要があります。

```python
trainer.train()  # トレーニングを開始し、W&B にログを記録する

# トレーニング後の分析、テスト、その他のログに記録されたコード

wandb.finish()
```

### 結果を可視化する

トレーニング結果をログに記録したら、[W&B Dashboard]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) で結果を動的に調べることができます。柔軟でインタラクティブな可視化により、一度に数十の run を簡単に比較したり、興味深い発見を拡大したり、複雑なデータから洞察を引き出したりできます。

## 高度な機能と FAQ

### 最適なモデルを保存するにはどうすればよいですか？
`Trainer` に渡される `TrainingArguments` で `load_best_model_at_end=True` が設定されている場合、W&B は最もパフォーマンスの高いモデルチェックポイントを Artifacts に保存します。

チーム全体で最適なモデルのバージョンを一元化して、ML タスクごとに整理したり、本番環境用にステージングしたり、さらなる評価のためにブックマークしたり、ダウンストリームの Model CI/CD プロセスを開始したりする場合は、モデルチェックポイントを Artifacts に保存していることを確認してください。Artifacts にログ記録されると、これらのチェックポイントは [モデルレジストリ]({{< relref path="/guides/models/registry/model_registry/" lang="ja" >}}) に昇格できます。

### 保存したモデルをロードするにはどうすればよいですか？

`WANDB_LOG_MODEL` を使用してモデルを W&B Artifacts に保存した場合、追加のトレーニングや推論を実行するためにモデルの重みをダウンロードできます。以前に使用したのと同じ Hugging Face アーキテクチャにロードするだけです。

```python
# 新しい run を作成する
with wandb.init(project="amazon_sentiment_analysis") as run:
    # Artifact の名前とバージョンを渡す
    my_model_name = "model-bert-base-high-lr:latest"
    my_model_artifact = run.use_artifact(my_model_name)

    # モデルの重みをフォルダーにダウンロードしてパスを返す
    model_dir = my_model_artifact.download()

    # 同じモデルクラスを使用して、そのフォルダーから Hugging Face モデルをロードする
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=num_labels
    )

    # 追加のトレーニングを行うか、推論を実行する
```

### チェックポイントからトレーニングを再開するにはどうすればよいですか？
`WANDB_LOG_MODEL='checkpoint'` を設定している場合は、`TrainingArguments` の `model_name_or_path` 引数として `model_dir` を使用し、`Trainer` に `resume_from_checkpoint=True` を渡すことで、トレーニングを再開することもできます。

```python
last_run_id = "xxxxxxxx"  # wandb ワークスペースから run_id を取得する

# run_id から wandb run を再開する
with wandb.init(
    project=os.environ["WANDB_PROJECT"],
    id=last_run_id,
    resume="must",
) as run:
    # Artifact を run に接続する
    my_checkpoint_name = f"checkpoint-{last_run_id}:latest"
    my_checkpoint_artifact = run.use_artifact(my_model_name)

    # チェックポイントをフォルダーにダウンロードしてパスを返す
    checkpoint_dir = my_checkpoint_artifact.download()

    # モデルとトレーナーを再初期化する
    model = AutoModelForSequenceClassification.from_pretrained(
        "<model_name>", num_labels=num_labels
    )
    # ここに優れたトレーニング引数があります。
    training_args = TrainingArguments()

    trainer = Trainer(model=model, args=training_args)

    # チェックポイントディレクトリを使用して、チェックポイントからトレーニングを再開するようにしてください
    trainer.train(resume_from_checkpoint=checkpoint_dir)
```

### トレーニング中に評価サンプルをログに記録して表示するにはどうすればよいですか

Transformers `Trainer` を介した W&B へのロギングは、Transformers ライブラリの [`WandbCallback`](https://huggingface.co/transformers/main_classes/callback.html#transformers.integrations.WandbCallback) によって処理されます。Hugging Face ロギングをカスタマイズする必要がある場合は、`WandbCallback` をサブクラス化し、Trainer クラスの追加のメソッドを活用する追加の機能を追加することで、このコールバックを変更できます。

以下は、この新しいコールバックを HF Trainer に追加するための一般的なパターンであり、さらに下には、評価出力を W&B Table にログ記録するためのコード完全な例があります。


```python
# Trainer を通常どおりにインスタンス化する
trainer = Trainer()

# 新しいロギングコールバックをインスタンス化し、Trainer オブジェクトを渡す
evals_callback = WandbEvalsCallback(trainer, tokenizer, ...)

# コールバックを Trainer に追加する
trainer.add_callback(evals_callback)

# Trainer のトレーニングを通常どおりに開始する
trainer.train()
```

#### トレーニング中に評価サンプルを表示する

次のセクションでは、`WandbCallback` をカスタマイズしてモデル予測を実行し、トレーニング中に評価サンプルを W&B Table にログ記録する方法を示します。Trainer コールバックの `on_evaluate` メソッドを使用して、`eval_steps` ごとに実行します。

ここでは、tokenizer を使用してモデル出力から予測とラベルをデコードする `decode_predictions` 関数を作成しました。

次に、予測とラベルから pandas DataFrame を作成し、DataFrame に `epoch` 列を追加します。

最後に、DataFrame から `wandb.Table` を作成し、wandb にログ記録します。
さらに、予測を `freq` エポックごとにログ記録することで、ログ記録の頻度を制御できます。

**注**: 通常の `WandbCallback` とは異なり、このカスタムコールバックは、`Trainer` の初期化中ではなく、`Trainer` のインスタンス化**後**にトレーナーに追加する必要があります。
これは、`Trainer` インスタンスが初期化中にコールバックに渡されるためです。

```python
from transformers.integrations import WandbCallback
import pandas as pd


def decode_predictions(tokenizer, predictions):
    labels = tokenizer.batch_decode(predictions.label_ids)
    logits = predictions.predictions.argmax(axis=-1)
    prediction_text = tokenizer.batch_decode(logits)
    return {"labels": labels, "predictions": prediction_text}


class WandbPredictionProgressCallback(WandbCallback):
    """トレーニング中にモデル予測をログ記録するカスタム WandbCallback。

    このコールバックは、トレーニング中の各ロギングステップでモデル予測とラベルを wandb.Table にログ記録します。これにより、
    トレーニングの進行状況に応じてモデル予測を可視化できます。

    属性：
        trainer (Trainer): Hugging Face Trainer インスタンス。
        tokenizer (AutoTokenizer): モデルに関連付けられた tokenizer。
        sample_dataset (Dataset): 予測を生成するための検証データセットのサブセット。
        num_samples (int, optional): 予測を生成するために検証データセットから選択するサンプル数。デフォルトは 100 です。
        freq (int, optional): ロギングの頻度。デフォルトは 2 です。
    """

    def __init__(self, trainer, tokenizer, val_dataset, num_samples=100, freq=2):
        """WandbPredictionProgressCallback インスタンスを初期化します。

        Args:
            trainer (Trainer): Hugging Face Trainer インスタンス。
            tokenizer (AutoTokenizer): モデルに関連付けられた tokenizer。
            val_dataset (Dataset): 検証データセット。
            num_samples (int, optional): 予測を生成するために検証データセットから選択するサンプル数。
              デフォルトは 100 です。
            freq (int, optional): ロギングの頻度。デフォルトは 2 です。
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset.select(range(num_samples))
        self.freq = freq

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        # 予測をログ記録することで、ログ記録の頻度を制御する
        # `freq` エポックごと
        if state.epoch % self.freq == 0:
            # 予測を生成する
            predictions = self.trainer.predict(self.sample_dataset)
            # 予測とラベルをデコードする
            predictions = decode_predictions(self.tokenizer, predictions)
            # 予測を wandb.Table に追加する
            predictions_df = pd.DataFrame(predictions)
            predictions_df["epoch"] = state.epoch
            records_table = self._wandb.Table(dataframe=predictions_df)
            # テーブルを wandb にログ記録する
            self._wandb.log({"sample_predictions": records_table})


# まず、Trainer をインスタンス化する
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
)

# WandbPredictionProgressCallback をインスタンス化する
progress_callback = WandbPredictionProgressCallback(
    trainer=trainer,
    tokenizer=tokenizer,
    val_dataset=lm_dataset["validation"],
    num_samples=10,
    freq=2,
)

# コールバックをトレーナーに追加する
trainer.add_callback(progress_callback)
```

詳細な例については、こちらの [colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Custom_Progress_Callback.ipynb) を参照してください。


### その他の W&B 設定はありますか？

環境変数を設定することで、`Trainer` でログ記録される内容をさらに構成できます。W&B 環境変数の完全なリストは、[こちらにあります]({{< relref path="/guides/hosting/env-vars/" lang="ja" >}})。

| 環境変数        | 使用法                                                                                                                                                                                                                                                                                                |
| -------------------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `WANDB_PROJECT`      | プロジェクトに名前を付ける (デフォルトは `huggingface`)                                                                                                                                                                                                                                                 |
| `WANDB_LOG_MODEL`    | <p>モデルチェックポイントを W&B Artifact としてログ記録する (デフォルトは `false`) </p><ul><li><code>false</code> (デフォルト): モデルチェックポイントなし </li><li><code>checkpoint</code>: チェックポイントは、args.save_steps ごとにアップロードされます (Trainer の TrainingArguments で設定)。 </li><li><code>end</code>: 最終的なモデルチェックポイントは、トレーニングの最後にアップロードされます。</li></ul>                                                                                                                                                                                                                                                                       |
| `WANDB_WATCH`        | <p>モデルの勾配、パラメーター、またはどちらもログに記録するかどうかを設定する</p><ul><li><code>false</code> (デフォルト): 勾配またはパラメーターのログ記録なし </li><li><code>gradients</code>: 勾配のヒストグラムをログ記録する </li><li><code>all</code>: 勾配とパラメーターのヒストグラムをログ記録する</li></ul> |
| `WANDB_DISABLED`     | ログ記録を完全にオフにするには、`true` に設定します (デフォルトは `false`)                                                                                                                                                                                                                                |
| `WANDB_SILENT`       | wandb によって出力される出力を非表示にするには、`true` に設定します (デフォルトは `false`)                                                                                                                                                                                                                   |

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


### `wandb.init` をカスタマイズするにはどうすればよいですか？

`Trainer` が使用する `WandbCallback` は、`Trainer` が初期化されるときに内部で `wandb.init` を呼び出します。`Trainer` が初期化される前に `wandb.init` を呼び出すことで、run を手動で設定することもできます。これにより、W&B run の構成を完全に制御できます。

`init` に渡す可能性のある例を次に示します。`wandb.init` の使用方法の詳細については、[リファレンスドキュメントを確認してください]({{< relref path="/ref/python/init.md" lang="ja" >}})。

```python
wandb.init(
    project="amazon_sentiment_analysis",
    name="bert-base-high-lr",
    tags=["baseline", "high-lr"],
    group="bert",
)
```


## 追加のリソース

以下は、Transformer と W&B に関連する 6 つの記事で、楽しめるかもしれません。

<details>

<summary>Hugging Face Transformers のハイパーパラメーター最適化</summary>

* Hugging Face Transformers のハイパーパラメーター最適化のための 3 つの戦略 (グリッド検索、ベイズ最適化、および Population Based Training) が比較されます。
* Hugging Face transformers から標準の uncased BERT モデルを使用し、SuperGLUE ベンチマークから RTE データセットでファインチューンします。
* 結果は、Population Based Training が Hugging Face transformer モデルのハイパーパラメーター最適化に最も効果的なアプローチであることを示しています。

完全なレポートは [こちら](https://wandb.ai/amogkam/transformers/reports/Hyperparameter-Optimization-for-Hugging-Face-Transformers--VmlldzoyMTc2ODI) をお読みください。
</details>

<details>

<summary>Hugging Tweets: ツイートを生成するモデルをトレーニングする</summary>

* この記事では、著者は、HuggingFace Transformer モデルを、たった5分で誰かのツイートでファインチューンする方法を説明します。
* モデルは、次のパイプラインを使用します。ツイートのダウンロード、データセットの最適化、初期実験、ユーザー間の損失の比較、モデルのファインチューニング。

完全なレポートは [こちら](https://wandb.ai/wandb/huggingtweets/reports/HuggingTweets-Train-a-Model-to-Generate-Tweets--VmlldzoxMTY5MjI) をお読みください。
</details>

<details>

<summary>Hugging Face BERT および WB を使用した文章分類</summary>

* この記事では、自然言語処理における最近のブレークスルーの力を活用して、文章分類子を作成し、NLP への転移学習の応用例に焦点を当てます。
* 単一文章分類には、言語的容認性 (CoLA) データセットを使用します。これは、2018 年 5 月に最初に公開された、文法的に正しいまたは正しくないとラベル付けされた文章のセットです。
* Google の BERT を使用して、さまざまな NLP タスクで最小限の労力で高性能モデルを作成します。

完全なレポートは [こちら](https://wandb.ai/cayush/bert-finetuning/reports/Sentence-Classification-With-Huggingface-BERT-and-W-B--Vmlldzo4MDMwNA) をお読みください。
</details>

<details>

<summary>Hugging Face モデルのパフォーマンスを追跡するためのステップバイステップガイド</summary>

* W&B と Hugging Face transformers を使用して、GLUE ベンチマークで DistilBERT をトレーニングします。これは、BERT より 40% 小さいが、BERT の精度の 97% を保持する Transformer です。
* GLUE ベンチマークは、NLP モデルをトレーニングするための 9 つのデータセットとタスクのコレクションです

完全なレポートは [こちら](https://wandb.ai/jxmorris12/huggingface-demo/reports/A-Step-by-Step-Guide-to-Tracking-HuggingFace-Model-Performance--VmlldzoxMDE2MTU) をお読みください。
</details>

<details>

<summary>HuggingFace における Early Stopping の例</summary>

* Early Stopping 正規化を使用して Hugging Face Transformer をファインチューンするには、PyTorch または TensorFlow でネイティブに実行できます。
* TensorFlow で EarlyStopping コールバックを使用することは、`tf.keras.callbacks.EarlyStopping` コールバックで簡単です。
* PyTorch には、既製の Early Stopping メソッドはありませんが、GitHub Gist で利用可能な実用的な Early Stopping フックがあります。

完全なレポートは [こちら](https://wandb.ai/ayush-thakur/huggingface/reports/Early-Stopping-in-HuggingFace-Examples--Vmlldzo0MzE2MTM) をお読みください。
</details>

<details>

<summary>カスタムデータセットで Hugging Face Transformers をファインチューンする方法</summary>

カスタム IMDB データセットで、感情分析 (バイナリ分類) 用の DistilBERT transformer をファインチューンします。

完全なレポートは [こちら](https://wandb.ai/ayush-thakur/huggingface/reports/How-to-Fine-Tune-HuggingFace-Transformers-on-a-Custom-Dataset--Vmlldzo0MzQ2MDc) をお読みください。
</details>

## ヘルプまたは機能のリクエスト

Hugging Face W&B インテグレーションに関する問題、質問、または機能のリクエストについては、[Hugging Face フォーラムのこのスレッド](https://discuss.huggingface.co/t/logging-experiment-tracking-with-w-b/498) に投稿するか、Hugging Face [Transformers GitHub リポジトリ](https://github.com/huggingface/transformers) で問題を提起してください。
