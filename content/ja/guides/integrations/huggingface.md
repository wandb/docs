---
title: Hugging Face Transformers
menu:
  default:
    identifier: huggingface
    parent: integrations
weight: 110
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Optimize_Hugging_Face_models_with_Weights_&_Biases.ipynb" >}}

[Hugging Face Transformers](https://huggingface.co/transformers/) ライブラリは、BERTのような最新のNLPモデルや mixed precision・gradient チェックポイントなどのトレーニング手法を簡単に利用できるようにします。[W&B インテグレーション](https://huggingface.co/transformers/main_classes/callback.html#transformers.integrations.WandbCallback) を追加することで、シンプルさを損なうことなく、柔軟でリッチな実験管理やモデルのバージョン管理をインタラクティブなダッシュボードから実現できます。

## 高度なログ機能をたった数行で

```python
os.environ["WANDB_PROJECT"] = "<my-amazing-project>"  # W&B Project 名を設定
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # 全てのモデルチェックポイントを記録

from transformers import TrainingArguments, Trainer

args = TrainingArguments(..., report_to="wandb")  # W&B ロギングを有効化
trainer = Trainer(..., args=args)
```
{{< img src="/images/integrations/huggingface_gif.gif" alt="HuggingFace dashboard" >}}

{{% alert %}}
すぐに動くサンプルコードを試すなら、この [Google Colab](https://wandb.me/hf) をご覧ください。
{{% /alert %}}

## まずは実験管理を始めよう

### サインアップとAPIキーの作成

APIキーはあなたのマシンをW&Bに認証します。APIキーはユーザープロファイルから発行できます。

{{% alert %}}
より簡単な方法として、[W&B認証ページ](https://wandb.ai/authorize) から直接APIキーを発行できます。表示されたAPIキーをコピーし、パスワードマネージャーなどの安全な場所に保存しましょう。
{{% /alert %}}

1. 画面右上のユーザープロフィールアイコンをクリックします。
1. **User Settings** を選択し、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックし、表示されたAPIキーをコピーします。APIキーを非表示にするにはページを再読み込みしてください。

### `wandb` ライブラリをインストールし、ログインする

ローカル環境で `wandb` ライブラリをインストールし、ログインするには:

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref "/guides/models/track/environment-variables.md" >}}) にAPIキーをセットします。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb` ライブラリをインストールし、ログインします。



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

初めてW&Bを使う場合は、[クイックスタート]({{< relref "/guides/quickstart.md" >}})もチェックしてみてください。

### プロジェクト名を設定する

W&B Project には、関連する run から記録されたすべてのチャート、データ、モデルが保存されます。プロジェクトに名前を付けることで作業を整理し、すべての情報をひとつの場所にまとめることができます。

run を Project に追加するには、`WANDB_PROJECT` 環境変数をプロジェクト名にセットするだけです。`WandbCallback` はこのプロジェクト名の環境変数を自動で検知し、run のセットアップ時に利用します。

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
`Trainer` を初期化 *する前に*、必ずプロジェクト名をセットしてください。
{{% /alert %}}

プロジェクト名を指定しない場合は、デフォルトで `huggingface` になります。

### トレーニング run をW&Bに記録する

これは **最も重要なステップ** です。`Trainer` のトレーニング引数を設定する際、コード内あるいはコマンドラインで `report_to` を `"wandb"` にセットして W&B のログ機能を有効にしてください。

`TrainingArguments` 内の `logging_steps` 引数で、トレーニング中にどのくらいの頻度でメトリクスをW&Bに送信するかを制御できます。また `run_name` 引数を使えばW&B上でrunに名前をつけることもできます。

これだけでOKです。モデルの損失、評価メトリクス、モデルトポロジー、勾配などがトレーニング中に自動でW&Bに記録されます。

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

```bash
python run_glue.py \     # Pythonスクリプトを実行
  --report_to wandb \    # W&Bへのロギングを有効にする
  --run_name bert-base-high-lr \   # W&B runの名前（任意）
  # 他のコマンドライン引数
```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```python
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    # 他の引数
    report_to="wandb",  # W&Bへのロギングを有効にする
    run_name="bert-base-high-lr",  # W&B上でrun名（任意）
    logging_steps=1,  # W&Bへのログ頻度
)

trainer = Trainer(
    # 他の引数
    args=args,  # トレーニング引数
)

trainer.train()  # トレーニングとW&Bへのログを開始
```

{{% /tab %}}
{{< /tabpane >}}

{{% alert %}}
TensorFlowを利用している場合、PyTorch の `Trainer` の代わりに TensorFlow の `TFTrainer` を使うだけでOKです。
{{% /alert %}}

### モデルのチェックポイントを有効にする

[Artifacts]({{< relref "/guides/core/artifacts/" >}}) を使えば、最大100GBまでモデルやデータセットを無料保存でき、さらにW&B [Registry]({{< relref "/guides/core/registry/" >}})と連携可能です。Registry を使うことで、モデル登録・閲覧・評価・ステージング・本番デプロイまでを一元管理できます。

Hugging Face モデルのチェックポイントを Artifacts に記録するには、`WANDB_LOG_MODEL` 環境変数を *いずれかひとつ* にセットしてください。

- **`checkpoint`**: [`TrainingArguments`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments) の `args.save_steps` ごとにチェックポイントをアップロード
- **`end`**: トレーニング終了時に `load_best_model_at_end` が有効ならモデルをアップロード
- **`false`**: モデルをアップロードしない


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

この設定以降に初期化したすべてのTransformers `Trainer`で、モデルはW&Bプロジェクトにアップロードされます。記録したモデルチェックポイントは [Artifacts]({{< relref "/guides/core/artifacts/" >}}) UIから確認でき、完全なモデルリネージも含まれます（[UIでのモデルチェックポイント例はこちら](https://wandb.ai/wandb/arttest/artifacts/model/iv3_trained/5334ab69740f9dda4fed/lineage?_gl=1*yyql5q*_ga*MTQxOTYyNzExOS4xNjg0NDYyNzk1*_ga_JH1SJHJQXJ*MTY5MjMwNzI2Mi4yNjkuMS4xNjkyMzA5NjM2LjM3LjAuMA..)）。

{{% alert %}}
デフォルトでは、`WANDB_LOG_MODEL` が `end` の場合は `model-{run_id}`、`checkpoint` の場合は `checkpoint-{run_id}` という名前でW&B Artifactsに保存されます。  
ただし、`TrainingArguments` に [`run_name`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.run_name) を指定している場合は `model-{run_name}` または `checkpoint-{run_name}` として保存されます。
{{% /alert %}}

#### W&B Registry
チェックポイントを Artifacts に記録したら、[Registry]({{< relref "/guides/core/registry/" >}}) を使い、優秀なモデルを登録してチーム内で一元管理できます。Registry では、タスク別にベストなモデル管理、モデルライフサイクル管理、MLOps全体の追跡や監査、さらには[オートメーション]({{< relref "/guides/core/automations/" >}})による自動処理が可能です。

モデル Artifact のリンク方法は [Registry]({{< relref "/guides/core/registry/" >}}) を参照してください。

### トレーニング中の評価出力を可視化

トレーニングや評価時にモデルの出力を可視化するのは、モデル理解に非常に重要です。

Transformers Trainer のコールバックシステムを活用すると、例えばモデルのテキスト生成出力や予測結果など、追加の有用データをW&B テーブルにログできます。

W&B Tableへの評価出力の記録方法は、下記 [カスタムロギングセクション]({{< relref "#custom-logging-log-and-view-evaluation-samples-during-training" >}}) をご覧ください。

{{< img src="/images/integrations/huggingface_eval_tables.png" alt="Shows a W&B Table with evaluation outputs" >}}

### W&B Runを終了する（ノートブック向け）

Pythonスクリプトでトレーニングしている場合、スクリプト終了時に自動で W&B run も終了します。

Jupyter や Google Colabノートブックを利用している場合は、トレーニング終了時に `run.finish()` を呼び出して明示的にrun終了を伝えてください。

```python
run = wandb.init()
trainer.train()  # トレーニング＆W&Bロギング開始

# トレーニング後の分析やテスト等

run.finish()
```

### 結果を可視化する

トレーニングの結果を記録したら、[W&B ダッシュボード]({{< relref "/guides/models/track/workspaces.md" >}}) でダイナミックに分析できます。複数runの比較や詳細な可視化も簡単、結果を一元的・インタラクティブに理解できます。

## 発展機能とFAQ

### ベストモデルの保存方法

`Trainer` に `TrainingArguments` で `load_best_model_at_end=True` を渡すと、パフォーマンスが最も良いモデルチェックポイントが Artifacts に自動保存されます。

モデルチェックポイントを Artifacts として保存したら、[Registry]({{< relref "/guides/core/registry/" >}}) へプロモート可能です。Registry では下記ができます。
- タスク毎にベストなモデルバージョンを整理
- モデルをチーム内で一元管理・共有
- モデルの本番環境ステージングや評価用のブックマーク
- 下流のCI/CDプロセスを自動実行

### 保存モデルの読み込み方法

`WANDB_LOG_MODEL` を使ってモデルをW&B Artifactsに保存していた場合、追加トレーニングや推論用にモデルウェイトをダウンロードできます。いつでも、以前と同じ Hugging Face アーキテクチャにロード可能です。

```python
# 新しいrunを作成
with wandb.init(project="amazon_sentiment_analysis") as run:
    # Artifactの名前とバージョンを指定
    my_model_name = "model-bert-base-high-lr:latest"
    my_model_artifact = run.use_artifact(my_model_name)

    # モデルウェイトをディレクトリにダウンロードし、パスを返す
    model_dir = my_model_artifact.download()

    # 同じモデルクラスを使ってHugging Faceモデルをロード
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=num_labels
    )

    # 追加学習や推論など
```

### チェックポイントからトレーニングを再開したい

`WANDB_LOG_MODEL='checkpoint'` をセットしていた場合、上記のようにチェックポイントディレクトリを `TrainingArguments` の `model_name_or_path` 引数に、そのうえで `Trainer` に `resume_from_checkpoint=True` を渡せば再開可能です。

```python
last_run_id = "xxxxxxxx"  # wandb workspaceでrun_idを取得

# run_idからwandb runを再開
with wandb.init(
    project=os.environ["WANDB_PROJECT"],
    id=last_run_id,
    resume="must",
) as run:
    # Artifactをrunに接続
    my_checkpoint_name = f"checkpoint-{last_run_id}:latest"
    my_checkpoint_artifact = run.use_artifact(my_model_name)

    # チェックポイントをディレクトリにダウンロードし、パスを返す
    checkpoint_dir = my_checkpoint_artifact.download()

    # モデルとtrainerの再初期化
    model = AutoModelForSequenceClassification.from_pretrained(
        "<model_name>", num_labels=num_labels
    )
    # トレーニング引数をここで設定
    training_args = TrainingArguments()

    trainer = Trainer(model=model, args=training_args)

    # checkpointディレクトリからトレーニング再開
    trainer.train(resume_from_checkpoint=checkpoint_dir)
```

### トレーニング中に評価サンプルを記録・表示する

Transformers `Trainer` を利用したW&Bへのロギングは、Transformersライブラリ内の [`WandbCallback`](https://huggingface.co/transformers/main_classes/callback.html#transformers.integrations.WandbCallback) が担っています。Hugging Face のロギングをカスタマイズしたい場合は、このコールバックをサブクラスして `Trainer` の追加メソッドを活用可能です。

以下は、この新しいコールバックをHF Trainerに追加する一般的なパターンです。さらに下には評価サンプルをW&B Tableとしてロギングする完全なコード例を示します。

```python
# まず通常通りTrainerをインスタンス化
trainer = Trainer()

# Trainerオブジェクトを渡して新しいロギングコールバックを作成
evals_callback = WandbEvalsCallback(trainer, tokenizer, ...)

# Trainerにコールバックを追加
trainer.add_callback(evals_callback)

# 通常どおりTrainerでトレーニング開始
trainer.train()
```

#### トレーニング中に評価サンプルを表示する

次のセクションでは、`WandbCallback` をカスタマイズし、トレーニング中に `on_evaluate` メソッドを使ってモデル予測と評価サンプルをW&B Tableに記録する方法を紹介します。

ここでは `decode_predictions` 関数で、トークナイザーを使って予測とラベルをデコードします。

その後、予測やラベルからpandasのDataFrameを作り、エポック番号列を追加。

最後に `wandb.Table` を作成し、wandbに記録します。
ログの頻度（何エポックごとに記録するか）は `freq` で調整できます。

**注意**: 標準の `WandbCallback` とは異なり、このカスタムコールバックはTrainerの初期化「後」に追加してください。コールバック初期化時にTrainerインスタンスを渡す必要があります。

```python
from transformers.integrations import WandbCallback
import pandas as pd


def decode_predictions(tokenizer, predictions):
    labels = tokenizer.batch_decode(predictions.label_ids)
    logits = predictions.predictions.argmax(axis=-1)
    prediction_text = tokenizer.batch_decode(logits)
    return {"labels": labels, "predictions": prediction_text}


class WandbPredictionProgressCallback(WandbCallback):
    """トレーニング中にモデル予測をロギングするカスタム WandbCallback

    このコールバックはトレーニング進行中、各ロギングステップでモデル予測とラベルを wandb.Table として記録します。
    進行に伴うモデル予測の可視化が可能となります。

    Attributes:
        trainer (Trainer): Hugging Face Trainerのインスタンス
        tokenizer (AutoTokenizer): モデル用のトークナイザー
        sample_dataset (Dataset): 予測生成用に検証データセットから抜粋したサブセット
        num_samples (int, optional): 予測に使う検証データのサンプル数（デフォルト100）
        freq (int, optional): ロギングの頻度（デフォルト2）
    """

    def __init__(self, trainer, tokenizer, val_dataset, num_samples=100, freq=2):
        """WandbPredictionProgressCallback の初期化

        Args:
            trainer (Trainer): Hugging Face Trainer のインスタンス
            tokenizer (AutoTokenizer): モデル用トークナイザー
            val_dataset (Dataset): 検証データセット
            num_samples (int, optional): 予測に使う検証データサンプル数
            freq (int, optional): ロギングの頻度
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset.select(range(num_samples))
        self.freq = freq

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        # freqエポックごとに予測をロギング
        if state.epoch % self.freq == 0:
            # 予測生成
            predictions = self.trainer.predict(self.sample_dataset)
            # 予測とラベルをデコード
            predictions = decode_predictions(self.tokenizer, predictions)
            # wandb.Tableに追加
            predictions_df = pd.DataFrame(predictions)
            predictions_df["epoch"] = state.epoch
            records_table = self._wandb.Table(dataframe=predictions_df)
            # テーブルをwandbにロギング
            self._wandb.log({"sample_predictions": records_table})


# まずTrainerを用意
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
)

# WandbPredictionProgressCallbackを作成
progress_callback = WandbPredictionProgressCallback(
    trainer=trainer,
    tokenizer=tokenizer,
    val_dataset=lm_dataset["validation"],
    num_samples=10,
    freq=2,
)

# Trainerにコールバック追加
trainer.add_callback(progress_callback)
```

より詳細な例は [colabノートブック](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Custom_Progress_Callback.ipynb) もご参照ください。

### W&Bで利用できる追加設定

`Trainer` との連携時に、環境変数でさらに記録対象を細かく制御できます。W&B環境変数の全一覧は[こちら]({{< relref "/guides/hosting/env-vars.md" >}})。

| 環境変数            | 用途                                                                                                                                                                                                                                                                                                    |
| -------------------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `WANDB_PROJECT`      | プロジェクト名の指定（デフォルトは `huggingface`）                                                                                                                                                                                                                                                      |
| `WANDB_LOG_MODEL`    | <p>モデルチェックポイントをW&B Artifactとして記録（デフォルトは<code>false</code>） </p><ul><li><code>false</code> (デフォルト): チェックポイント記録しない </li><li><code>checkpoint</code>: TrainerのTrainingArgumentsで指定したsave_stepsごとにアップロード</li><li><code>end</code>: トレーニング終了時に最終モデルをアップロード</li></ul>                                                                   |
| `WANDB_WATCH`        | <p>モデルの勾配・パラメータのヒストグラムを記録するか指定</p><ul><li><code>false</code> (デフォルト): ログしない </li><li><code>gradients</code>: 勾配ヒストグラムを記録 </li><li><code>all</code>: 勾配・パラメータ両方のヒストグラムを記録</li></ul> |
| `WANDB_DISABLED`     | `true` でW&Bロギングを完全オフ（デフォルトは `false`） |
| `WANDB_QUIET`        | `true` で重要な出力のみ標準出力に表示（デフォルトは `false`）                                                                                                                                                                                                                                     |
| `WANDB_SILENT`       | `true` でwandbの出力を完全に抑制（デフォルトは `false`）                                                                                                                                                                                                                                |

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


### `wandb.init` をカスタマイズしたい場合

`Trainer` が利用する `WandbCallback` は、初期化時に内部で `wandb.init` を呼びます。run構成を手動で詳細にセットアップしたい場合は、`Trainer` 初期化前に明示的に `wandb.init` を呼び出せば、W&Bのrun設定をフルにコントロールできます。

`init` に渡す主な例は下記。詳細は[`wandb.init()`のリファレンス]({{< relref "/ref/python/sdk/functions/init.md" >}})をご覧ください。

```python
wandb.init(
    project="amazon_sentiment_analysis",
    name="bert-base-high-lr",
    tags=["baseline", "high-lr"],
    group="bert",
)
```

## 関連リソース

以下はTransformersとW&Bに関連したおすすめ記事6選です。

<details>

<summary>Hugging Face Transformersのハイパーパラメータ最適化</summary>

* Hugging Face Transformers向けのハイパーパラメータ最適化について、グリッド検索、ベイズ最適化、Population Based Training の3手法を比較します。
* 標準のuncased BERTモデルをSuperGLUE ベンチマークのRTEデータセットでファインチューニングします。
* 結果として、Population Based Trainingがハイパーパラメータ最適化で最も効果的でした。

[Hyperparameter Optimization for Hugging Face Transformers report を読む](https://wandb.ai/amogkam/transformers/reports/Hyperparameter-Optimization-for-Hugging-Face-Transformers--VmlldzoyMTc2ODI)。
</details>

<details>

<summary>Hugging Tweets: Tweet生成モデルをトレーニングする</summary>

* 著者はGPT2 HuggingFace Transformerを任意の人物のツイートデータで5分程度でファインチューニングする方法を紹介します。
* パイプラインはTweetのダウンロード、データセット最適化、初期実験、ユーザ間の損失比較、モデルのファインチューニング。

[フルレポートはこちら](https://wandb.ai/wandb/huggingtweets/reports/HuggingTweets-Train-a-Model-to-Generate-Tweets--VmlldzoxMTY5MjI)。
</details>

<details>

<summary>Sentence Classification With Hugging Face BERT and WB</summary>

* ここでは最新NLP技術を活用した文分類用の高性能モデルを構築するTransfer Learning応用例を紹介します。
* 単文分類にはCoLAデータセット（文の妥当性ラベル付き）を使用します。
* GoogleのBERTを使ってNLPタスクを最小の労力で高性能化します。

[フルレポートはこちら](https://wandb.ai/cayush/bert-finetuning/reports/Sentence-Classification-With-Huggingface-BERT-and-W-B--Vmlldzo4MDMwNA)。
</details>

<details>

<summary>Hugging Faceモデルパフォーマンス追跡ガイド</summary>

* W&BとHugging Face transformersでDistilBERT（BERTより40%小型・97%精度）をGLUEベンチマーク向けにトレーニングします。
* GLUEベンチマークはNLPモデルの訓練用データセット・タスク集です。

[フルレポートはこちら](https://wandb.ai/jxmorris12/huggingface-demo/reports/A-Step-by-Step-Guide-to-Tracking-HuggingFace-Model-Performance--VmlldzoxMDE2MTU)。
</details>

<details>

<summary>HuggingFaceでのEarly Stopping実例</summary>

* Early Stopping正則化付きのファインチューニングは、PyTorchでもTensorFlowでも可能です。
* TensorFlowでは `tf.keras.callbacks.EarlyStopping` コールバックが標準で使えます。
* PyTorchには標準EarlyStoppingはありませんが、GitHub Gistに動作サンプルあり。

[フルレポートはこちら](https://wandb.ai/ayush-thakur/huggingface/reports/Early-Stopping-in-HuggingFace-Examples--Vmlldzo0MzE2MTM)。
</details>

<details>

<summary>独自データセット上でのHugging Face Transformersファインチューニング方法</summary>

DistilBERTトランスフォーマを独自のIMDBデータセットでセンチメント分析（二値分類）タスクにファインチューニングします。

[フルレポートはこちら](https://wandb.ai/ayush-thakur/huggingface/reports/How-to-Fine-Tune-HuggingFace-Transformers-on-a-Custom-Dataset--Vmlldzo0MzQ2MDc)。
</details>

## サポート・機能リクエスト

Hugging Face W&Bインテグレーションについて、質問や要望があれば[Hugging Faceフォーラムのこちらのスレッド](https://discuss.huggingface.co/t/logging-experiment-tracking-with-w-b/498)や、Hugging Face [Transformers GitHubリポジトリ](https://github.com/huggingface/transformers)のissueへお気軽にご投稿ください。