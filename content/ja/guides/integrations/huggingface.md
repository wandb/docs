---
title: Hugging Face Transformers
menu:
  default:
    identifier: ja-guides-integrations-huggingface
    parent: integrations
weight: 110
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Optimize_Hugging_Face_models_with_Weights_&_Biases.ipynb" >}}

[Hugging Face Transformers](https://huggingface.co/transformers/) ライブラリは、BERT のような最先端の NLP モデルや混合精度、勾配チェックポイントのようなトレーニング技術を簡単に使用可能にします。[W&B インテグレーション](https://huggingface.co/transformers/main_classes/callback.html#transformers.integrations.WandbCallback) は、インタラクティブで中央集中型のダッシュボードに豊かで柔軟な実験管理とモデルバージョン管理を追加しながら、その使いやすさを損なうことはありません。

## 数行で可能な次世代のロギング

```python
os.environ["WANDB_PROJECT"] = "<my-amazing-project>"  # W&B プロジェクトの名前を指定
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # すべてのモデルチェックポイントをログ

from transformers import TrainingArguments, Trainer

args = TrainingArguments(..., report_to="wandb")  # W&B ロギングをオン
trainer = Trainer(..., args=args)
```
{{< img src="/images/integrations/huggingface_gif.gif" alt="W&B インタラクティブダッシュボードで実験結果を探索" >}}

{{% alert %}}
すぐにワーキングコードに飛び込みたい場合は、この [Google Colab](https://wandb.me/hf) をチェックしてください。
{{% /alert %}}

## 始めましょう: 実験を管理する

### サインアップと APIキー の作成

APIキー は、あなたのマシンを W&B に認証します。APIキー はユーザープロファイルから生成できます。

{{% alert %}}
より簡潔なアプローチを求める場合は、[https://wandb.ai/authorize](https://wandb.ai/authorize) に直接アクセスして APIキー を生成できます。表示される APIキー をコピーし、パスワードマネージャーのような安全な場所に保存してください。
{{% /alert %}}

1. 右上のユーザープロファイルアイコンをクリックします。
2. **ユーザー設定** を選択し、**APIキー** セクションまでスクロールします。
3. **表示** をクリックします。表示される APIキー をコピーします。APIキー を非表示にするには、ページをリロードします。

### `wandb` ライブラリをインストールしてログインする

`wandb` ライブラリをローカルにインストールしてログインするには、次の手順を実行します。

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) をあなたの API キーに設定します。

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

{{% tab header="Python ノートブック" value="python" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

初めて W&B を使う場合は、[**クイックスタート**]({{< relref path="/guides/quickstart.md" lang="ja" >}})をチェックすると良いでしょう。

### プロジェクトの名前を付ける

W&B プロジェクトは、関連する run からログしたすべてのチャート、データ、およびモデルが保存される場所です。プロジェクトに名前を付けることで、作業を整理し、単一のプロジェクトに関するすべての情報を一箇所にまとめることができます。

run をプロジェクトに追加するには、単に `WANDB_PROJECT` 環境変数をプロジェクトの名前に設定します。`WandbCallback` は、このプロジェクト名の環境変数を拾い上げ、run のセットアップ時に使用します。

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

{{% tab header="Python ノートブック" value="notebook" %}}

```notebook
%env WANDB_PROJECT=amazon_sentiment_analysis
```

{{% /tab %}}

{{< /tabpane >}}

{{% alert %}}
プロジェクト名は `Trainer` を初期化する*前*に設定することを確認してください。
{{% /alert %}}

プロジェクト名が指定されていない場合、プロジェクト名はデフォルトで `huggingface` になります。

### トレーニング run を W&B にログする

これは、`Trainer` トレーニング引数を定義する際の**最も重要なステップ**です。コード内、またはコマンドラインから `report_to` を `"wandb"` に設定することで、W&B でのログが有効になります。

`TrainingArguments` の `logging_steps` 引数は、トレーニングメトリクスがトレーニング中に W&B にプッシュされる頻度を制御します。また、`run_name` 引数を使用して W&B のトレーニング run に名前を付けることもできます。

ここまでで完了です。これで、モデルはトレーニング中に損失、評価メトリクス、モデルトポロジー、勾配を W&B にログします。

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

```bash
python run_glue.py \     # Pythonスクリプトを実行
  --report_to wandb \    # W&B へのログを有効化
  --run_name bert-base-high-lr \   # W&B run の名前 (オプション)
  # その他のコマンドライン引数
```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```python
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    # 他の引数とキーワード引数
    report_to="wandb",  # W&B へのログを有効化
    run_name="bert-base-high-lr",  # W&B run の名前 (オプション)
    logging_steps=1,  # W&B へのログ頻度
)

trainer = Trainer(
    # 他の引数とキーワード引数
    args=args,  # トレーニング引数
)

trainer.train()  # トレーニングを開始し W&B にログ
```

{{% /tab %}}
{{< /tabpane >}}

{{% alert %}}
TensorFlow を使用していますか？PyTorch の `Trainer` を TensorFlow の `TFTrainer` に置き換えるだけです。
{{% /alert %}}

### モデルのチェックポイントをオンにする 

W&B の [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) を使用して、最大 100GB までのモデルやデータセットを無料で保存できます。その後、W&B の [Model Registry]({{< relref path="/guides/models/registry/model_registry/" lang="ja" >}}) を使ってモデルを登録し、ステージングやプロダクション環境でのデプロイメントの準備を行います。

Hugging Face モデルチェックポイントを Artifacts にログするには、`WANDB_LOG_MODEL` 環境変数を `end`、`checkpoint`、または `false` のいずれかに設定します：

- **`checkpoint`**: [`TrainingArguments`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments) の `args.save_steps` ごとにチェックポイントがアップロードされます。
- **`end`**: トレーニング終了時にモデルがアップロードされます。

`WANDB_LOG_MODEL` を `load_best_model_at_end` と組み合わせて、トレーニング終了時に最良のモデルをアップロードします。

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

{{% tab header="Python ノートブック" value="notebook" %}}

```notebook
%env WANDB_LOG_MODEL="checkpoint"
```

{{% /tab %}}

{{< /tabpane >}}

ここから、すべての Transformers `Trainer` を初期化すると、モデルが W&B プロジェクトにアップロードされます。ログしたモデルチェックポイントは、[Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) UI から確認でき、完全なモデルリネージを含みます（例として UI からモデルチェックポイントを確認 [こちら](https://wandb.ai/wandb/arttest/artifacts/model/iv3_trained/5334ab69740f9dda4fed/lineage?_gl=1*yyql5q*_ga*MTQxOTYyNzExOS4xNjg0NDYyNzk1*_ga_JH1SJHJQXJ*MTY5MjMwNzI2Mi4yNjkuMS4xNjkyMzA5NjM2LjM3LjAuMA..)）。

{{% alert %}}
デフォルトでは、`WANDB_LOG_MODEL` が `end` に設定されるとモデルは `model-{run_id}` として W&B Artifacts に保存され、`checkpoint` に設定されると `checkpoint-{run_id}` として保存されます。ただし、`TrainingArguments` に [`run_name`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.run_name) を渡すと、モデルは `model-{run_name}` または `checkpoint-{run_name}` として保存されます。
{{% /alert %}}

#### W&B モデルレジストリ
チェックポイントを Artifacts にログした後は、**[Model Registry]({{< relref path="/guides/models/registry/model_registry/" lang="ja" >}})** を使用して、最良のモデルチェックポイントを登録し、チーム全体で集中管理できます。ここでは、タスクごとに最高のモデルを整理し、モデルライフサイクルを管理し、ML ライフサイクル全体での簡単なトラッキングと監査を促進し、Webhook やジョブで下流のアクションを[オートメーション]({{< relref path="/guides/models/automations/project-scoped-automations/#create-a-webhook-automation" lang="ja" >}})できます。

モデルアーティファクトをモデルレジストリにリンクする方法については、[Model Registry]({{< relref path="/guides/models/registry/model_registry/" lang="ja" >}}) ドキュメントを参照してください。

### トレーニング中の評価結果を可視化

トレーニングや評価中にモデルの出力を可視化することは、モデルのトレーニングがどのように進行しているかを本当に理解するためにしばしば重要です。

Transformers Trainer のコールバックシステムを使用することで、モデルのテキスト生成出力やその他の予測を W&B テーブルにログするなど、追加の役立つデータを W&B にログできます。

トレーニング中に評価出力をログして W&B テーブルのようにログする方法について、**[カスタムロギングセクション]({{< relref path="#custom-logging-log-and-view-evaluation-samples-during-training" lang="ja" >}})** を以下で確認してください。

{{< img src="/images/integrations/huggingface_eval_tables.png" alt="W&B テーブルで評価出力を表示" >}}

### W&B Run を終了する（ノートブックのみ）

トレーニングが Python スクリプト内にカプセル化されている場合、スクリプトが終了すると W&B run は終了します。

Jupyter や Google Colab ノートブックを使用している場合、`wandb.finish()` を呼び出してトレーニングが終了したことを知らせる必要があります。

```python
trainer.train()  # トレーニングを開始し W&B にログ

# ポストトレーニング分析、テスト、その他のログ済みコード

wandb.finish()
```

### 結果を可視化する

トレーニング結果がログされたら、[W&B ダッシュボード]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}})で動的に結果を探索できます。多数の run を一度に比較し、興味深い学びにズームインし、柔軟でインタラクティブな可視化を使って複雑なデータからインサイトを得ることができます。

## 高度な機能と FAQ

### 最高のモデルはどのように保存しますか？
`TrainingArguments` で `load_best_model_at_end=True` が設定されている場合、W&B は Artifacts に最高の性能を持つモデルチェックポイントを保存します。

チーム全体で最高のモデルバージョンを中央集約化し、ML タスクで整理し、プロダクションにステージングし、さらなる評価のためにブックマークしたり、下流のモデル CI/CD プロセスを開始したりしたい場合は、モデルチェックポイントを必ず Artifacts に保存してください。Artifacts にログされたら、これらのチェックポイントを [Model Registry]({{< relref path="/guides/models/registry/model_registry/" lang="ja" >}}) に昇格させることができます。

### 保存したモデルをロードする方法は？

`WANDB_LOG_MODEL` を使用してモデルを W&B Artifacts に保存した場合、追加のトレーニングや推論を実行するためにモデルの重みをダウンロードできます。同じ Hugging Face アーキテクチャに戻してロードするだけです。

```python
# 新しい run を作成
with wandb.init(project="amazon_sentiment_analysis") as run:
    # アーティファクトの名前とバージョンを指定
    my_model_name = "model-bert-base-high-lr:latest"
    my_model_artifact = run.use_artifact(my_model_name)

    # モデルの重みをフォルダにダウンロードし、パスを返す
    model_dir = my_model_artifact.download()

    # Hugging Face モデルをそのフォルダからロード
    # 同じモデルクラスを使用
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=num_labels
    )

    # 追加のトレーニングまたは推論を実行
```

### チェックポイントからトレーニングを再開する方法は？
`WANDB_LOG_MODEL='checkpoint'` を設定していた場合、`model_dir` を `TrainingArguments` の `model_name_or_path` 引数として使用し、`resume_from_checkpoint=True` を `Trainer` に渡すことで、チェックポイントからトレーニングを再開できます。

```python
last_run_id = "xxxxxxxx"  # wandb ワークスペースから run_id を取得

# run_id から wandb run を再開
with wandb.init(
    project=os.environ["WANDB_PROJECT"],
    id=last_run_id,
    resume="must",
) as run:
    # アーティファクトを run に接続
    my_checkpoint_name = f"checkpoint-{last_run_id}:latest"
    my_checkpoint_artifact = run.use_artifact(my_model_name)

    # フォルダにチェックポイントをダウンロードし、パスを返す
    checkpoint_dir = my_checkpoint_artifact.download()

    # モデルとトレーナーを再初期化
    model = AutoModelForSequenceClassification.from_pretrained(
        "<model_name>", num_labels=num_labels
    )
    # トレーニング引数
    training_args = TrainingArguments()

    trainer = Trainer(model=model, args=training_args)

    # チェックポイントからトレーニングを再開するためにチェックポイントディレクトリを使用することを確認
    trainer.train(resume_from_checkpoint=checkpoint_dir)
```

### トレーニング中に評価サンプルをログして表示する方法は？

Transformers ライブラリの [`WandbCallback`](https://huggingface.co/transformers/main_classes/callback.html#transformers.integrations.WandbCallback) を介して W&B へのロギングは処理されます。Hugging Face のロギングをカスタマイズする必要がある場合は、`WandbCallback` をサブクラス化して `Trainer` クラスの追加メソッドを利用する機能を追加できます。

以下に、この新しいコールバックを HF トレーナーに追加する一般的なパターンを示します。評価出力を W&B テーブルにログするための完全なコード例は、さらに下に示されています：

```python
# 通常どおりトレーナーをインスタンス化
trainer = Trainer()

# トレーナーオブジェクトを渡して新しいロギングコールバックをインスタンス化
evals_callback = WandbEvalsCallback(trainer, tokenizer, ...)

# コールバックをトレーナーに追加
trainer.add_callback(evals_callback)

# 通常どおりトレーナートレーニングを開始
trainer.train()
```

#### トレーニング中に評価サンプルを表示

以下のセクションでは、`WandbCallback` をカスタマイズしてモデル予測を実行し、トレーニング中に W&B テーブルに評価サンプルをログする方法を示します。`Trainer` コールバックの `on_evaluate` メソッドを使用して、すべての `eval_steps` をログします。

ここでは、トークナイザーを使用してモデルの出力から予測とラベルをデコードする `decode_predictions` 関数を書きました。

次に、予測とラベルから pandas DataFrame を作成し、DataFrame に `epoch` 列を追加します。

最後に、DataFrame から `wandb.Table` を作成し、wandb にログします。
また、`freq` エポックごとに予測をログすることでログの頻度を制御できます。

**注**: 通常の `WandbCallback` とは異なり、このカスタムコールバックは `Trainer` の初期化時ではなく、`Trainer` がインスタンス化された後にトレーナーに追加する必要があります。これは、`Trainer` インスタンスがコールバックの初期化時に渡されるためです。

```python
from transformers.integrations import WandbCallback
import pandas as pd


def decode_predictions(tokenizer, predictions):
    labels = tokenizer.batch_decode(predictions.label_ids)
    logits = predictions.predictions.argmax(axis=-1)
    prediction_text = tokenizer.batch_decode(logits)
    return {"labels": labels, "predictions": prediction_text}


class WandbPredictionProgressCallback(WandbCallback):
    """トレーニング中のモデル予測をログするカスタム WandbCallback。

    このコールバックは、トレーニング中の各ログステップでモデル予測とラベルを wandb.Table にログします。
    トレーニングの進捗に合わせてモデル予測を視覚化できます。

    Attributes:
        trainer (Trainer): Hugging Face Trainer インスタンス。
        tokenizer (AutoTokenizer): モデルに関連付けられたトークナイザー。
        sample_dataset (Dataset): 予測を生成するための検証データセットのサブセット。
        num_samples (int, optional): 予測を生成するために検証データセットから選択するサンプル数。デフォルトは 100。
        freq (int, optional): ロギングの頻度。デフォルトは 2。
    """

    def __init__(self, trainer, tokenizer, val_dataset, num_samples=100, freq=2):
        """WandbPredictionProgressCallback インスタンスを初期化。

        Args:
            trainer (Trainer): Hugging Face Trainer インスタンス。
            tokenizer (AutoTokenizer): モデルに関連付けられたトークナイザー。
            val_dataset (Dataset): 検証データセット。
            num_samples (int, optional): 予測を生成するために検証データセットから選択するサンプル数。デフォルトは 100。
            freq (int, optional): ロギングの頻度。デフォルトは 2。
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset.select(range(num_samples))
        self.freq = freq

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        # ロギングの頻度を調整し、`freq` エポックごとに予測をログする
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


# まずトレーナーをインスタンス化
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
)

# WandbPredictionProgressCallback をインスタンス化
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

より詳細な例については、こちらの [colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Custom_Progress_Callback.ipynb) を参照してください。

### どのような追加の W&B 設定が利用可能ですか？

`Trainer` でログされる内容のさらなる設定は、環境変数を設定することで可能です。W&B 環境変数の詳細なリストは [こちら]({{< relref path="/guides/hosting/env-vars/" lang="ja" >}}) にあります。

| 環境変数 | 用途                                                                                                                                                                                                                                                                                                    |
| ---------------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `WANDB_PROJECT`      | プロジェクトに名前を付ける（デフォルトは `huggingface`）                                                                                                                                                                                                                                                          |
| `WANDB_LOG_MODEL`    | <p>モデルチェックポイントを W&B アーティファクトとしてログする（デフォルトは `false`）</p><ul><li><code>false</code>（デフォルト）：モデルのチェックポイントはありません</li><li><code>checkpoint</code>: Trainer の TrainingArguments で設定された args.save_steps ごとにチェックポイントがアップロードされます。</li><li><code>end</code>: トレーニング終了時に最終的なモデルチェックポイントがアップロードされます。</li></ul>                                                                                                                                                                                                                                  |
| `WANDB_WATCH`        | <p>モデルの勾配やパラメータをログするかどうかを設定</p><ul><li><code>false</code>（デフォルト）：勾配やパラメータのログはありません</li><li><code>gradients</code>: 勾配のヒストグラムをログ</li><li><code>all</code>: 勾配とパラメータのヒストグラムをログ</li></ul> |
| `WANDB_DISABLED`     | ロギングを完全にオフにするには `true` に設定（デフォルトは `false`）                                                                                                                                                                                                                                            |
| `WANDB_SILENT`       | wandb によって出力されるものをサイレンスするために `true` に設定（デフォルトは `false`）                                                                                                                                                                                                                                |

{{< tabpane text=true >}}

{{% tab header="コマンドライン" value="cli" %}}

```bash
WANDB_WATCH=all
WANDB_SILENT=true
```

{{% /tab %}}

{{% tab header="ノートブック" value="notebook" %}}

```notebook
%env WANDB_WATCH=all
%env WANDB_SILENT=true
```

{{% /tab %}}

{{< /tabpane >}}

### `wandb.init` をどのようにカスタマイズしますか？

`Trainer` が使用する `WandbCallback` は、`Trainer` が初期化されるときに内部で `wandb.init` を呼び出します。代わりに、`Trainer` が初期化される前に `wandb.init` を呼び出して run を手動で設定することもできます。これにより、W&B run の設定を完全に制御できます。

`init` に渡すものの例を以下に示します。`wandb.init` の使い方の詳細については、[リファレンスドキュメントをチェックしてください]({{< relref path="/ref/python/init.md" lang="ja" >}})。

```python
wandb.init(
    project="amazon_sentiment_analysis",
    name="bert-base-high-lr",
    tags=["baseline", "high-lr"],
    group="bert",
)
```

## 追加の資料

以下は、Transformers と W&B に関連する記事6つです。

<details>

<summary>Hugging Face Transformers のハイパーパラメーター最適化</summary>

* Hugging Face Transformers のハイパーパラメーター最適化の 3 つの戦略が比較されています：グリッド検索、ベイズ最適化、Pop/Pobulation Based Training。
* Hugging Face transformers からの標準の無条件 BERT モデルを使用し、SuperGLUE ベンチマークからの RTE データセットでファインチューンしたいと考えています。
* 結果は、Pop/Pobulation Based Training が Hugging Face transformer モデルのハイパーパラメーター最適化に最も効果的であることを示しています。

全文を読むには [こちら](https://wandb.ai/amogkam/transformers/reports/Hyperparameter-Optimization-for-Hugging-Face-Transformers--VmlldzoyMTc2ODI)。
</details>

<details>

<summary>変なつぶやきをハグ：ツイートを生成するモデルをトレーニングする</summary>

* 記事では、GPT2 HuggingFace Transformer モデルのプリトレインモデルにファインチューンする方法を 5 分で説明しています。
* モデルは以下のパイプラインを使用します：ツイートのダウンロード、データセットの最適化、初期実験、ユーザー間の損失の比較、モデルのファインチューン。

全文を読むには [こちら](https://wandb.ai/wandb/huggingtweets/reports/HuggingTweets-Train-a-Model-to-Generate-Tweets--VmlldzoxMTY5MjI)。
</details>

<details>

<summary>Hugging Face BERT と WB を使用した文分類</summary>

* この記事では、自然言語処理における最近のブレークスルーの力を利用して文分類を構築し、NLP における転移学習の応用に焦点を当てます。
* 単一文分類用の The Corpus of Linguistic Acceptability (CoLA) データセットを使用します。これは、文法的に正しいまたは間違っているとラベル付けされた文章のセットで、2018 年 5 月に初めて公開されました。
* Google's BERT を使用して、最小限の労力で幅広い NLP タスクで高性能なモデルを作成します。

全文を読むには [こちら](https://wandb.ai/cayush/bert-finetuning/reports/Sentence-Classification-With-Huggingface-BERT-and-W-B--Vmlldzo4MDMwNA)。
</details>

<details>

<summary>Hugging Face モデルのパフォーマンスをトラッキングするステップバイステップガイド</summary>

* W&B と Hugging Face transformers を使用して、DistilBERT をトレーニングします。DistilBERT は BERT より 40% 小さいが、BERT の精度の 97% を保持する Transformer で、GLUE ベンチマーク上でトレーニングします。
* GLUE ベンチマークは、NLP モデルをトレーニングするための 9 つのデータセットとタスクのコレクションです。

全文を読むには [こちら](https://wandb.ai/jxmorris12/huggingface-demo/reports/A-Step-by-Step-Guide-to-Tracking-HuggingFace-Model-Performance--VmlldzoxMDE2MTU)。
</details>

<details>

<summary>HuggingFace における早期停止の例</summary>

* Early Stopping 正則化を使用して Hugging Face Transformer をファインチューンすることは、PyTorch や TensorFlow でネイティブに実行できます。
* TensorFlow での EarlyStopping コールバックの使用は、`tf.keras.callbacks.EarlyStopping` コールバックを用いることで簡単です。
* PyTorch には、オフ・ザ・シェルフの早期停止メソッドはありませんが、GitHub Gist には動作する早期停止フックが利用可能です。

全文を読むには [こちら](https://wandb.ai/ayush-thakur/huggingface/reports/Early-Stopping-in-HuggingFace-Examples--Vmlldzo0MzE2MTM)。
</details>

<details>

<summary>Hugging Face Transformers をカスタムデータセットにファインチューンする方法</summary>

カスタム IMDB データセットでセンチメント分析（二値分類）を行うために、DistilBERT トランスフォーマーをファインチューンしています。

全文を読むには [こちら](https://wandb.ai/ayush-thakur/huggingface/reports/How-to-Fine-Tune-HuggingFace-Transformers-on-a-Custom-Dataset--Vmlldzo0MzQ2MDc)。
</details>

## ヘルプを求めるまたは機能をリクエストする

Hugging Face W&B インテグレーションに関する問題、質問、または機能リクエストがある場合は、[Hugging Face フォーラムのこのスレッド](https://discuss.huggingface.co/t/logging-experiment-tracking-with-w-b/498)に投稿するか、Hugging Face [Transformers GitHub リポジトリ](https://github.com/huggingface/transformers)に問題をオープンしてください。