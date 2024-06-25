---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';


# Hugging Face Transformers

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Optimize_Hugging_Face_models_with_Weights_&_Biases.ipynb"></CTAButtons>

[Hugging Face Transformers](https://huggingface.co/transformers/) ライブラリは、BERTのような最先端のNLPモデルや、mixed precisionやgradient checkpointingといったトレーニング技術を簡単に利用できるようにします。[W&B integration](https://huggingface.co/transformers/main_classes/callback.html#transformers.integrations.WandbCallback)は、柔軟で豊富な実験管理とモデルのバージョン管理をインタラクティブな中央ダッシュボードに追加し、その使いやすさを損なうことなく提供します。

## 🤗 数行で次世代のログ記録

```python
os.environ["WANDB_PROJECT"] = "<my-amazing-project>"  # W&Bプロジェクトの名前を設定
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # 全てのモデルチェックポイントをログ

from transformers import TrainingArguments, Trainer

args = TrainingArguments(..., report_to="wandb")  # W&Bログをオンに
trainer = Trainer(..., args=args)
```
![W&Bのインタラクティブダッシュボードで実験結果を探索](@site/static/images/integrations/huggingface_gif.gif)

:::info
すぐに動作するコードを試したい場合は、この[Google Colab](https://wandb.me/hf)をチェックしてください。
:::

## 始めましょう：実験を追跡

### 1) サインアップし、`wandb`ライブラリをインストールしてログイン

a) 無料アカウントに[**サインアップ**](https://wandb.ai/site)します

b) `wandb`ライブラリをpipインストールします

c) トレーニングスクリプトでログインするには、www.wandb.aiでアカウントにサインインし、[**Authorize page**](https://wandb.ai/authorize)で**APIキーを見つけてください**。

Weights and Biasesを初めて使用する場合は、[**quickstart**](../../quickstart.md)を確認してください。

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

[Project](../app/pages/project-page.md)は、関連するrunsから記録された全てのチャート、データ、およびモデルが保存される場所です。プロジェクトに名前を付けると、作業を整理し、単一プロジェクトに関する全ての情報を一箇所にまとめるのに役立ちます。

プロジェクトにrunを追加するには、単に`WANDB_PROJECT`環境変数をプロジェクトの名前に設定します。`WandbCallback`はこのプロジェクト名の環境変数を拾い上げ、runを設定する際に使用します。

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
必ず`Trainer`を初期化する前にプロジェクト名を設定してください。
:::

プロジェクト名が指定されていない場合、プロジェクト名はデフォルトで「huggingface」になります。

### 3) トレーニングRunsをW&Bにログ

これは**最も重要なステップ**です。`Trainer`トレーニング引数をコード内またはコマンドラインから設定する際に、`report_to`を`"wandb"`に設定してWeights & Biasesでのログ記録を有効にすることです。

`TrainingArguments`の`logging_steps`引数は、トレーニング中にどのくらいの頻度でW&Bにトレーニングメトリクスをプッシュするかを制御します。また、`run_name`引数を使用してW&Bでのトレーニングrunに名前を付けることもできます。

これで完了です！トレーニング中にモデルの損失、評価メトリクス、モデルのトポロジー、および勾配がWeights & Biasesに記録されます。

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python', value: 'python'},
    {label: 'Command Line', value: 'cli'},
  ]}>
  <TabItem value="cli">

```bash
python run_glue.py \     # Pythonスクリプトを実行
  --report_to wandb \    # W&Bへのロギングを有効に
  --run_name bert-base-high-lr \   # W&Bのrun名（任意）
  # 他のコマンドライン引数をここに
```

  </TabItem>
  <TabItem value="python">

```python
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    # 他のargsやkwargsをここに
    report_to="wandb",  # W&Bへのロギングを有効に
    run_name="bert-base-high-lr",  # W&Bのrun名（任意）
    logging_steps=1,  # W&Bにどのくらい頻繁にログを記録するか
)

trainer = Trainer(
    # 他のargsやkwargsをここに
    args=args,  # トレーニング引数を設定
)

trainer.train()  # トレーニングを開始しW&Bにログ
```

  </TabItem>
</Tabs>

:::info
TensorFlowを使用していますか？ PyTorchの`Trainer`をTensorFlowの`TFTrainer`に置き換えるだけです。
:::

### 4) モデルチェックポイントの有効化

Weights & Biasesの[Artifacts](../artifacts)を使用すると、最大100GBのモデルやデータセットを無料で保存でき、Weights & Biasesの[Model Registry](../model_registry)を使用して、モデルをステージングやプロダクション環境でのデプロイメントに向けて登録できます。

Hugging FaceモデルのチェックポイントをArtifactsにログするには、`WANDB_LOG_MODEL`環境変数を`end`、`checkpoint`、または`false`のいずれかに設定します：

- **`checkpoint`**: チェックポイントは[`TrainingArguments`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments)から`args.save_steps`ごとにアップロードされます。
- **`end`**: モデルはトレーニングの終了時にアップロードされます。

`WANDB_LOG_MODEL`と`load_best_model_at_end`を併用して、トレーニング終了時に最良のモデルをアップロードします。

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

今後初期化される任意のTransformers `Trainer`は、モデルをW&Bプロジェクトにアップロードします。ログされたモデルのチェックポイントは[Artifacts](../artifacts)のUIから閲覧でき、完全なモデルリネージが含まれます（UIでのモデルチェックポイントの例は[こちら](https://wandb.ai/wandb/arttest/artifacts/model/iv3_trained/5334ab69740f9dda4fed/lineage?_gl=1*yyql5q*_ga*MTQxOTYyNzExOS4xNjg0NDYyNzk1*_ga_JH1SJHJQXJ*MTY5MjMwNzI2Mi4yNjkuMS4xNjkyMzA5NjM2LjM3LjAuMA..)）。

:::info
デフォルトでは、`WANDB_LOG_MODEL`が`end`または`checkpoint`に設定されている場合、あなたのモデルは`model-{run_id}`としてW&B Artifactsに保存されます。しかし、`TrainingArguments`に[`run_name`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.run_name)を渡すと、モデルは`model-{run_name}`または`checkpoint-{run_name}`として保存されます。
:::

#### W&B Model Registry
チェックポイントをArtifactsにログした後、Weights & Biasesの**[Model Registry](../model_registry)**を使用して最良のモデルチェックポイントを登録し、チーム全体で集中管理できます。ここで、タスクごとに最良のモデルを整理し、モデルのライフサイクルを管理し、MLライフサイクル全体で簡単な追跡と監査を支援し、ウェブフックやジョブと共に下流のアクションを[自動化](https://docs.wandb.ai/guides/models/automation)できます。

モデルArtifactをModel Registryにリンクする方法については[Model Registry](../model_registry)のドキュメントを参照してください。

### 5) トレーニング中の評価出力を可視化

トレーニング中や評価中にモデルの出力を可視化することは、モデルのトレーニング状況を確実に理解するために重要です。

Transformers Trainerのコールバックシステムを使用することで、W&B Tablesにモデルのテキスト生成出力や他の予測をログするなど、追加の役立つデータをW&Bにログできます。

トレーニング中に評価出力をログする方法については、以下の**[Custom logging section](#custom-logging-log-and-view-evaluation-samples-during-training)**を参照してください。次のようなW&Bテーブルに評価サンプルをログする方法がガイドされています：

![評価出力を表示するW&Bテーブル](/images/integrations/huggingface_eval_tables.png)

### 6) W&B Runを終了する（Notebookのみ）

トレーニングがPythonスクリプトにカプセル化されている場合は、スクリプトが終了するとW&Bのrunも終了します。

JupyterやGoogle Colabノートブックを使用している場合は、トレーニングが終了したことを`wandb.finish()`を呼び出して知らせる必要があります。

```python
trainer.train()  # W&Bにログしながらトレーニング開始

# トレーニング後の分析、テスト、他のログコード

wandb.finish()
```

### 7) 結果を可視化

トレーニング結果をログしたら、[W&B Dashboard](../track/app.md)で結果を動的に探索できます。複数のrunを一度に比較したり、興味深い知見にズームインしたり、柔軟でインタラクティブな可視化を用いて複雑なデータから洞察を引き出すのが簡単です。

## 高度な機能とFAQ

### 最高のモデルを保存するにはどうすればよいですか？
`Trainer`に渡される`TrainingArguments`に`load_best_model_at_end=True`を設定している場合、W&Bは最高性能のモデルチェックポイントをArtifactsに保存します。

チーム全体で最良のモデルバージョンを中央に集中化し、MLタスクごとに整理し、プロダクション用にステージングし、さらなる評価のためにブックマークしたり、下流のモデルCI/CDプロセスを開始したりするには、モデルチェックポイントをArtifactsに保存することを確認してください。ログされたチェックポイントは[Model Registry](../model_registry/intro.md)にプロモートすることができます。

### 保存されたモデルの読み込み

`WANDB_LOG_MODEL`を使用してモデルをW&B Artifactsに保存した場合、追加のトレーニングや推論を行うためにモデルの重みをダウンロードして読み込むことができます。同じHugging Faceアーキテクチャーに読み戻すだけです。

```python
# 新しいrunを作成
with wandb.init(project="amazon_sentiment_analysis") as run:
    # 値とバージョンのArtifactを渡します
    my_model_name = "model-bert-base-high-lr:latest"
    my_model_artifact = run.use_artifact(my_model_name)

    # フォルダにモデルの重みをダウンロードしパスを返します
    model_dir = my_model_artifact.download()

    # 同じモデルクラスを使ってHugging Faceモデルをそのフォルダから読み込みます
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=num_labels
    )

    # 追加のトレーニングや推論を行います
```

### チェックポイントからのトレーニング再開 
`WANDB_LOG_MODEL='checkpoint'`を設定していた場合、`model_dir`を`TrainingArguments`の`model_name_or_path`引数として使用し、`resume_from_checkpoint=True`を`Trainer`に渡すことでトレーニングを再開できます。

```python
last_run_id = "xxxxxxxx"  # wandb workspaceからrun_idを取得

# run_idからwandb runを再開
with wandb.init(
    project=os.environ["WANDB_PROJECT"],
    id=last_run_id,
    resume="must",
) as run:
    # 走にArtifactを接続
    my_checkpoint_name = f"checkpoint-{last_run_id}:latest"
    my_checkpoint_artifact = run.use_artifact(my_model_name)

    # チェックポイントをフォルダーにダウンロードしパスを返します
    checkpoint_dir = my_checkpoint_artifact.download()

    # モデルとトレーナーを再初期化
    model = AutoModelForSequenceClassification.from_pretrained(
        "<model_name>", num_labels=num_labels
    )
    # トレーニングの引数をここに設定
    training_args = TrainingArguments()

    trainer = Trainer(model=model, args=training_args)

    # チェックポイントのディレクトリを使用してトレーニングをチェックポイントから再開
    trainer.train(resume_from_checkpoint=checkpoint_dir)
```

### カスタムログ：トレーニング中に評価サンプルをログし表示

Transformersライブラリの`Trainer`を介してWeights & Biasesにログすることは、`WandbCallback`によってカバーされています。Hugging Faceのログ設定をカスタマイズする必要がある場合は、`WandbCallback`をサブクラス化して、Trainerクラスの追加メソッドを活用する機能を追加します。

以下は、この新しいコールバックをHF Trainerに追加する一般的なパターンであり、その下にはトレーニング中に評価出力をW&Bテーブルにログするためのコード完全な例があります：

```python
# 通常通りTrainerをインスタンス化
trainer = Trainer()

# Trainerオブジェクトを渡して新しいログコールバックをインスタンス化
evals_callback = WandbEvalsCallback(trainer, tokenizer, ...)

# コールバックをTrainerに追加
trainer.add_callback(evals_callback)

# 通常通りTrainerのトレーニングを開始
trainer.train()
```

#### トレーニング中に評価サンプルを表示

以下のセクションでは、`WandbCallback`をカスタマイズして、モデルの予測を実行し、トレーニング中に評価サンプルをW&Bテーブルにログする方法を示します。`on_evaluate`メソッドを使用して、`eval_steps`ごとに予測とラベルをデコードします。

ここでは、トークナイザーを使用してモデル出力から予測とラベルをデコードするための`decode_predictions`関数を作成しました。

次に、予測とラベルからpandas DataFrameを作成し、エポック列をDataFrameに追加します。

最後に、DataFrameから`wandb.Table`を作成し、wandbにログします。
さらに、エポックごとに`freq`ごとに予測をログする頻度を制御できます。

**注**: 通常の`WandbCallback`とは異なり、このカスタムコールバックは**Trainerが初期化された後**でTrainerに追加する必要があります。
これは、Trainerインスタンスが