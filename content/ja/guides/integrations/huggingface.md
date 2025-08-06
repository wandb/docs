---
title: Hugging Face Transformers
menu:
  default:
    identifier: ja-guides-integrations-huggingface
    parent: integrations
weight: 110
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Optimize_Hugging_Face_models_with_Weights_&_Biases.ipynb" >}}

[Hugging Face Transformers](https://huggingface.co/transformers/) ライブラリは、BERT などの最先端 NLP モデルや、混合精度学習・勾配チェックポイント機能などのトレーニング手法を簡単に扱うことができます。[W&B integration](https://huggingface.co/transformers/main_classes/callback.html#transformers.integrations.WandbCallback) を使えば、高機能かつ柔軟な実験管理やモデルのバージョン管理を、使いやすさを損なうことなく、インタラクティブな集中管理型ダッシュボードで実現できます。

## 数行で次世代のログを実現

```python
os.environ["WANDB_PROJECT"] = "<my-amazing-project>"  # W&Bプロジェクト名を設定
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # すべてのモデルチェックポイントをログ

from transformers import TrainingArguments, Trainer

args = TrainingArguments(..., report_to="wandb")  # W&B ロギングを有効化
trainer = Trainer(..., args=args)
```
{{< img src="/images/integrations/huggingface_gif.gif" alt="HuggingFace ダッシュボード" >}}

{{% alert %}}
すぐに動作するコードを試したい方は、[Google Colab](https://wandb.me/hf) をご覧ください。
{{% /alert %}}

## 使い始める：実験をトラッキングする

### サインアップして APIキー を作成

APIキー はお使いのマシンを W&B に認証するためのものです。ユーザープロフィールから APIキー を発行できます。

{{% alert %}}
より簡単な方法として、[W&B 認証ページ](https://wandb.ai/authorize) に直接アクセスして API キーを発行できます。表示された API キーをコピーして、パスワードマネージャーなど安全な場所に保存してください。
{{% /alert %}}

1. 画面右上のユーザープロフィールアイコンをクリックします。
1. **User Settings** を選択後、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックし、表示された APIキー をコピーします。APIキー を隠したい場合はページをリロードしてください。

### `wandb` ライブラリのインストールとログイン

ローカル環境に `wandb` ライブラリをインストールしてログインします。

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) をご自身の APIキー に設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb` ライブラリのインストールとログインを実行します。

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

初めて W&B をご利用の場合は、[クイックスタート]({{< relref path="/guides/quickstart.md" lang="ja" >}}) もご参考ください。

### プロジェクト名をつける

W&B Project とは、関連する Run から記録されたすべてのチャートやデータ、モデルが保存される場所です。プロジェクトに名前をつけておくことで、1 つのプロジェクトに関するすべての情報を整理しやすくなります。

Run を Project に追加するには、`WANDB_PROJECT` 環境変数に Project 名をセットします。`WandbCallback` がこの環境変数を自動的に読み取り、Run 設定時に使用します。

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
`Trainer` を初期化する _前_ にプロジェクト名を設定してください。
{{% /alert %}}

プロジェクト名を指定しない場合、プロジェクト名はデフォルトで `huggingface` になります。

### トレーニング Run を W&B に記録する

**これが一番大事なステップです。** `Trainer` のトレーニング引数をコードやコマンドラインで定義する際、`report_to` を `"wandb"` にすることで W&B へのログが有効になります。

`TrainingArguments` の `logging_steps` 引数で、トレーニング中にどのくらいの頻度でトレーニングメトリクスが W&B に送信されるかを制御します。さらに、`run_name` 引数で、W&B 上のトレーニング Run に名前をつけることもできます。

これだけです。これでトレーニング中の損失値や評価メトリクス、モデルの構造や勾配情報などが W&B に自動で記録されます。

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

```bash
python run_glue.py \     # Pythonスクリプトを実行
  --report_to wandb \    # W&Bへのログを有効化
  --run_name bert-base-high-lr \   # W&B Run名（オプション）
  # その他のコマンドライン引数
```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```python
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    # その他の引数・キーワード引数
    report_to="wandb",  # W&Bへのロギングを有効化
    run_name="bert-base-high-lr",  # W&B Run名（オプション）
    logging_steps=1,  # どのくらいの頻度でW&Bにログするか
)

trainer = Trainer(
    # その他の引数・キーワード引数
    args=args,  # トレーニング引数
)

trainer.train()  # トレーニング＆W&Bへのロギング開始
```

{{% /tab %}}
{{< /tabpane >}}

{{% alert %}}
TensorFlow を使う場合は、PyTorch の `Trainer` を TensorFlow の `TFTrainer` に置き換えてください。
{{% /alert %}}

### モデルのチェックポイントを記録する

[Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) を使えば、100GB までのモデルやデータセットを無料で保存できます。また、W&B の [Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) も活用できます。Registry ではモデルの登録や検証、ステージングや本番環境へのデプロイが可能です。

Hugging Face のモデルチェックポイントを Artifacts へ記録したい場合、`WANDB_LOG_MODEL` 環境変数を _1つ_ 設定します：

- **`checkpoint`**: [`TrainingArguments`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments) の `args.save_steps` で指定したタイミング毎にチェックポイントをアップロード
- **`end`**: トレーニング終了時にモデルをアップロード（`load_best_model_at_end` を設定時）
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

これ以降に初期化した Transformer `Trainer` は自動的にモデルを W&B プロジェクトにアップロードします。記録されたモデルチェックポイントは [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) の UI から確認でき、フルモデルリネージを含みます（UI上のモデルチェックポイント例は[こちら](https://wandb.ai/wandb/arttest/artifacts/model/iv3_trained/5334ab69740f9dda4fed/lineage?_gl=1*yyql5q*_ga*MTQxOTYyNzExOS4xNjg0NDYyNzk1*_ga_JH1SJHJQXJ*MTY5MjMwNzI2Mi4yNjkuMS4xNjkyMzA5NjM2LjM3LjAuMA..) ）。

{{% alert %}}
デフォルトでは、`WANDB_LOG_MODEL` が `end` の場合は `model-{run_id}`、`checkpoint` の場合は `checkpoint-{run_id}` という名前で Artifacts に保存されます。
ただし、`TrainingArguments` の [`run_name`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.run_name) を指定した場合は、`model-{run_name}` か `checkpoint-{run_name}` となります。
{{% /alert %}}

#### W&B Registry
チェックポイントを Artifacts に記録したら、[Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) を使って最良のモデルチェックポイントを管理・チームで集中管理できます。Registry を使えば、タスクごとにベストモデルを整理したり、モデルのライフサイクルを管理したり、ML の全ライフサイクルを追跡・監査し、[オートメーション]({{< relref path="/guides/core/automations/" lang="ja" >}}) で後工程も自動化できます。

モデル Artifacts のリンク方法は [Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) をご覧ください。
 
### トレーニング中に評価出力を可視化する

トレーニングや評価時のモデル出力をリアルタイムで可視化することで、モデルの学習状況をより深く理解できます。

Transformers Trainer のコールバック機能を活用すれば、W&B に、たとえばモデルのテキスト生成結果や予測値などの追加データを W&B Tables へログすることも可能です。

具体的な方法は、下記 [カスタムログセクション]({{< relref path="#custom-logging-log-and-view-evaluation-samples-during-training" lang="ja" >}}) で解説しています。W&B Table に記録された評価サンプルのイメージもご覧いただけます。

{{< img src="/images/integrations/huggingface_eval_tables.png" alt="W&B Tableに評価出力が表示されている例" >}}

### W&B Run の終了宣言（Notebookの場合のみ）

トレーニングが Python スクリプト内に収まっている場合、スクリプト終了と同時に W&B Run も終了します。

Jupyter や Google Colab ノートブックでトレーニングを行う場合は、`run.finish()` を呼び出してトレーニング終了を明示してください。

```python
run = wandb.init()
trainer.train()  # トレーニング＆W&Bへのロギング開始

# 学習後の分析・テスト・他のログ出力等

run.finish()
```

### 結果の可視化

トレーニング結果をログしたら、[W&B ダッシュボード]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) でダイナミックに分析できます。十数回の Run を一度に比較したり、気になる学びを深掘りしたり、柔軟なインタラクティブ可視化で複雑なデータからインサイトを発掘できます。

## 高度な機能・FAQ

### ベストモデルの保存方法は？

`Trainer` に `load_best_model_at_end=True` をセットした `TrainingArguments` を渡すと、W&B は最良のモデルチェックポイントを Artifacts に保存します。

モデルチェックポイントを Artifacts として保存していれば、それを [Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) に昇格させることもできます。Registry でできること：

- タスク別にベストモデルのバージョンを整理
- モデルの集中管理やチーム共有
- モデルをプロダクション環境用にステージング or さらなる評価用にブックマーク
- 下流のCI/CDプロセスのトリガー

### 保存したモデルをロードするには？

W&B Artifacts に `WANDB_LOG_MODEL` でモデルを保存したら、追加トレーニングや推論用にモデル重みをダウンロードできます。同じ Hugging Face アーキテクチャにロードしてください。

```python
# 新しいrunを作成
with wandb.init(project="amazon_sentiment_analysis") as run:
    # Artifactの名前とバージョンを指定
    my_model_name = "model-bert-base-high-lr:latest"
    my_model_artifact = run.use_artifact(my_model_name)

    # モデル重みをダウンロード・パスを返す
    model_dir = my_model_artifact.download()

    # 対応するモデルクラスでロード
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=num_labels
    )

    # 追加トレーニング・推論等
```

### チェックポイントからトレーニングを再開するには？

`WANDB_LOG_MODEL='checkpoint'` を使っていれば、`model_dir` を `TrainingArguments` の `model_name_or_path` 引数に使い、`Trainer` の `resume_from_checkpoint=True` でトレーニングを再開できます。

```python
last_run_id = "xxxxxxxx"  # wandb workspace から run_id を取得

# run_id からW&B Run を再開
with wandb.init(
    project=os.environ["WANDB_PROJECT"],
    id=last_run_id,
    resume="must",
) as run:
    # Run に Artifact を接続
    my_checkpoint_name = f"checkpoint-{last_run_id}:latest"
    my_checkpoint_artifact = run.use_artifact(my_model_name)

    # チェックポイントをダウンロードし、パスを取得
    checkpoint_dir = my_checkpoint_artifact.download()

    # モデル・Trainerを再初期化
    model = AutoModelForSequenceClassification.from_pretrained(
        "<model_name>", num_labels=num_labels
    )
    # トレーニング引数を用意
    training_args = TrainingArguments()

    trainer = Trainer(model=model, args=training_args)

    # checkpoint_dir を使ってトレーニング再開
    trainer.train(resume_from_checkpoint=checkpoint_dir)
```

### トレーニング中に評価サンプルをログ・表示する方法

Transformers `Trainer` での W&B ロギングは、Transformers ライブラリ内の [`WandbCallback`](https://huggingface.co/transformers/main_classes/callback.html#transformers.integrations.WandbCallback) により行われています。より細かくロギング内容を制御したい場合は、`WandbCallback` をサブクラス化して Trainer クラスの追加メソッドと連携するカスタムコールバックを作成できます。

HF Trainer へ新しいコールバックを追加するパターン例、および評価出力を W&B Table へログする完全なコード例は下記です：

```python
# 普通通りTrainerを初期化
trainer = Trainer()

# 新しいロギング用コールバックを作成（Trainerオブジェクトを渡す）
evals_callback = WandbEvalsCallback(trainer, tokenizer, ...)

# コールバックをTrainerに追加
trainer.add_callback(evals_callback)

# 通常通りトレーニング開始
trainer.train()
```

#### トレーニング中の評価サンプル表示

下記コードは `WandbCallback` をカスタマイズし、モデルの予測と評価サンプルをトレーニング中に W&B Table へログする例です。`on_evaluate` メソッドで `eval_steps` 毎に実行します。

まず `decode_predictions` 関数で、トークナイザーを使って予測値とラベルをデコードします。

次に、pandas の DataFrame に変換し、エポック情報も追加します。

最後に、`wandb.Table` を作成して wandb へログします。`freq` 引数でロギング頻度も制御可能です。

**注意:** このカスタムコールバックは `Trainer` インスタンス生成 _後_ にコールバックを追加する必要があります。これは Trainer インスタンスをコールバック作成時に渡すためです。

```python
from transformers.integrations import WandbCallback
import pandas as pd


def decode_predictions(tokenizer, predictions):
    labels = tokenizer.batch_decode(predictions.label_ids)
    logits = predictions.predictions.argmax(axis=-1)
    prediction_text = tokenizer.batch_decode(logits)
    return {"labels": labels, "predictions": prediction_text}


class WandbPredictionProgressCallback(WandbCallback):
    """トレーニング中にモデルの予測値をログするカスタムWandbCallback

    このコールバックはトレーニング時のモデル予測値とラベルをwandb.Tableへ記録します。
    トレーニング進捗に合わせて予測値を可視化することができます。

    属性:
        trainer (Trainer): Hugging Face Trainerインスタンス
        tokenizer (AutoTokenizer): モデル用トークナイザー
        sample_dataset (Dataset): 検証データセットのサブセット
        num_samples (int, optional): ログ用サンプル数（デフォルト100）
        freq (int, optional): ログ頻度（デフォルト2）
    """

    def __init__(self, trainer, tokenizer, val_dataset, num_samples=100, freq=2):
        """WandbPredictionProgressCallback インスタンスの初期化

        引数:
            trainer (Trainer): Hugging Face Trainerインスタンス
            tokenizer (AutoTokenizer): モデル用トークナイザー
            val_dataset (Dataset): 検証データセット
            num_samples (int, optional): サンプル数（デフォルト100）
            freq (int, optional): ログ頻度（デフォルト2）
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset.select(range(num_samples))
        self.freq = freq

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        # freqエポック毎に予測値をログ
        if state.epoch % self.freq == 0:
            # 予測値生成
            predictions = self.trainer.predict(self.sample_dataset)
            # 予測値とラベルをデコード
            predictions = decode_predictions(self.tokenizer, predictions)
            # wandb.Tableへ追加
            predictions_df = pd.DataFrame(predictions)
            predictions_df["epoch"] = state.epoch
            records_table = self._wandb.Table(dataframe=predictions_df)
            # wandbにテーブルをログ
            self._wandb.log({"sample_predictions": records_table})


# まずTrainerを初期化
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

# コールバックをTrainerへ追加
trainer.add_callback(progress_callback)
```

より詳細な例は [こちらのcolab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Custom_Progress_Callback.ipynb) も参照ください。

### W&B で利用可能な追加設定は？

`Trainer` でログする内容のさらなる詳細調整は、環境変数の設定で行えます。W&B で使用可能な全環境変数一覧は [こちら]({{< relref path="/guides/hosting/env-vars.md" lang="ja" >}}) をご覧ください。

| 環境変数              | 用途                                                                                                                                                                                                                                                                                                      |
| -------------------- |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `WANDB_PROJECT`      | プロジェクト名を指定（デフォルト `huggingface`）                                                                                                                                                                                                                                                         |
| `WANDB_LOG_MODEL`    | <p>モデルチェックポイントを W&B Artifacts にログ（デフォルト `false`） </p><ul><li><code>false</code>（デフォルト）：モデルチェックポイントの記録なし</li><li><code>checkpoint</code>：Trainer の TrainingArguments の args.save_steps ごとにチェックポイントをアップ</li><li><code>end</code>：トレーニング終了時に最終チェックポイントをアップ</li></ul>                        |
| `WANDB_WATCH`        | <p>モデルの勾配・パラメータのログ設定</p><ul><li><code>false</code>（デフォルト）：勾配・パラメータのログなし</li><li><code>gradients</code>：勾配のヒストグラムをログ</li><li><code>all</code>：勾配＋パラメータのヒストグラムをログ</li></ul>                          |
| `WANDB_DISABLED`     | `true` で全ログを無効化（デフォルト`false`） |
| `WANDB_QUIET`.       | `true` で標準出力を重要な内容のみに制限（デフォルト`false`）                                                                                                                                                                                                                                              |
| `WANDB_SILENT`       | `true` で wandb の出力を全て消す（デフォルト`false`）                                                                                                                                                                                                             |

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


### `wandb.init` をカスタマイズするには？

`Trainer` が利用する `WandbCallback` は内部的に `wandb.init` を呼びますが、独自に Run を細かく制御したい場合は、`Trainer` 初期化前に自分で `wandb.init` を呼ぶことも可能です。

主な `init` の使い方は下記の例です。`wandb.init()` の詳細は[公式リファレンス]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) を参照ください。

```python
wandb.init(
    project="amazon_sentiment_analysis",
    name="bert-base-high-lr",
    tags=["baseline", "high-lr"],
    group="bert",
)
```

## 追加リソース

下記は Transformers や W&B に関するおすすめ記事 6選です。

<details>

<summary>Hugging Face Transformers のハイパーパラメータ最適化</summary>

* Hugging Face Transformers のハイパーパラメータ最適化手法として、「グリッドサーチ」「ベイズ最適化」「Population Based Training」の3つを比較。
* 標準のUncased BERTモデル（Hugging Face transformers）を使い、SuperGLUEベンチマークのRTEデータセットでファインチューニング。
* 結果、Population Based Training がハイパーパラメータ最適化で最も効果的であることがわかりました。

[Hyperparameter Optimization for Hugging Face Transformers report](https://wandb.ai/amogkam/transformers/reports/Hyperparameter-Optimization-for-Hugging-Face-Transformers--VmlldzoyMTc2ODI) を読む
</details>

<details>

<summary>Hugging Tweets: ツイート生成モデルをトレーニングする</summary>

* 本記事では、事前学習済みのGPT2 HuggingFace Transformerモデルを好きなユーザーのツイートに対して5分でファインチューニングする手順を紹介。
* パイプライン：ツイートのダウンロード→データセット最適化→初回実験→ユーザー毎の損失比較→モデルのファインチューニング。

[記事全文はこちら](https://wandb.ai/wandb/huggingtweets/reports/HuggingTweets-Train-a-Model-to-Generate-Tweets--VmlldzoxMTY5MjI)
</details>

<details>

<summary>Sentence Classification With Hugging Face BERT and WB</summary>

* 本記事では、NLP の最新成果を活用した文分類器を作成し、転移学習の NLP への応用に焦点を当てます。
* 単文分類には CoLA (The Corpus of Linguistic Acceptability) データセットを使い、これは正しい・誤った文法の単文を集めたもの（2018年初出）。
* Google の BERT を用いて、最小限の努力で高性能な NLP モデルを構築します。

[記事全文はこちら](https://wandb.ai/cayush/bert-finetuning/reports/Sentence-Classification-With-Huggingface-BERT-and-W-B--Vmlldzo4MDMwNA)
</details>

<details>

<summary>Hugging Face モデル性能の追跡ガイド</summary>

* W&B と Hugging Face transformers を使って、BERT より 40% 小さいが 97% の正確性を持つ DistilBERT を GLUE ベンチマークでトレーニングします。
* GLUE ベンチマークは、NLP モデル学習向けの9種のデータセットとタスクを集めたものです。

[記事全文はこちら](https://wandb.ai/jxmorris12/huggingface-demo/reports/A-Step-by-Step-Guide-to-Tracking-HuggingFace-Model-Performance--VmlldzoxMDE2MTU)
</details>

<details>

<summary>HuggingFace での Early Stopping 活用例</summary>

* Hugging Face Transformer のファインチューニングで Early Stopping 正則化を利用（PyTorch または TensorFlow）。
* TensorFlow の EarlyStopping コールバック利用は `tf.keras.callbacks.EarlyStopping` で簡単に実装可能。
* PyTorch では既製の early stopping メソッドは無いが、GitHub Gist で利用可能なフックがあります。

[記事全文はこちら](https://wandb.ai/ayush-thakur/huggingface/reports/Early-Stopping-in-HuggingFace-Examples--Vmlldzo0MzE2MTM)
</details>

<details>

<summary>独自データセットで Hugging Face Transformers をファインチューニングする方法</summary>

カスタム IMDB データセット（二値分類）でセンチメント分析用 DistilBERT トランスフォーマーをファインチューニングします。

[記事全文はこちら](https://wandb.ai/ayush-thakur/huggingface/reports/How-to-Fine-Tune-HuggingFace-Transformers-on-a-Custom-Dataset--Vmlldzo0MzQ2MDc)
</details>

## サポート・機能要望

Hugging Face W&B integration に対するご質問や要望・不具合報告などがある場合は、[Hugging Face フォーラムのこのスレッド](https://discuss.huggingface.co/t/logging-experiment-tracking-with-w-b/498) または Hugging Face [Transformers GitHubリポジトリ](https://github.com/huggingface/transformers) で issue をお送りください。