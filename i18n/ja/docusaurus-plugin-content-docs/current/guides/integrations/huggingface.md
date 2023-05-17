import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Hugging Faceトランスフォーマー

[Hugging Faceトランスフォーマー](https://huggingface.co/transformers/) ライブラリは、BERTのような最先端のNLPモデルや、混合精度演算や勾配チェックポイントのようなトレーニング手法を簡単に利用できるようにしています。[W&Bインテグレーション](https://huggingface.co/transformers/main\_classes/callback.html#transformers.integrations.WandbCallback)は、使いやすさを損なうことなく、インタラクティブな中央集約ダッシュボードに豊富で柔軟な実験トラッキングとモデルバージョン管理を追加します。

## 🤗 2行で次世代のログ作成

```python
from transformers import TrainingArguments, Trainer

args = TrainingArguments(... , report_to="wandb")
trainer = Trainer(... , args=args)
```
![W&Bのインタラクティブダッシュボードで実験結果を探索](@site/static/images/integrations/huggingface_gif.gif)

## このガイドで解説する内容

* W&BとHugging Face トランスフォーマーを組み合わせて、NLP の実験を追跡する方法[**基本的な使い方**](huggingface.md#getting-started-track-and-save-your-models)
* [**W&B Hugging Face 統合の高度な機能を利用する方法**](../track/intro.md) を詳しく紹介し、実験のトラッキングを最大限に活用する方法。

:::info
すぐに実際のコードを試してみたい場合は、この[Google Colab](https://wandb.me/hf)をチェックしてください。
:::

## はじめに：実験の追跡

### 1) サインアップし、`wandb` ライブラリをインストールしてログインする

a) [**サインアップ**](https://wandb.ai/site)して無料アカウントを取得
b) `wandb`ライブラリをpipでインストール

c) トレーニングスクリプトでログインするには、www.wandb.aiでアカウントにサインインする必要があり、その後、 [**Authorize page**](https://wandb.ai/authorize) **でAPIキーを見つけることができます。**

Weights and Biasesを初めて使う場合は、[**クイックスタート**](../../quickstart.md) をチェックしてみてください。

<Tabs
  defaultValue="cli"
  values={[
    {label: 'コマンドライン', value: 'cli'},
    {label: 'ノートブック', value: 'notebook'},
  ]}>
  <TabItem value="cli">

```python
pip install wandb

wandb login
```

  </TabItem>
  <TabItem value="notebook">

```python
!pip install wandb

import wandb
wandb.login()
```
</TabItem>
</Tabs>

### 2) プロジェクトの名前を付ける

[プロジェクト](../app/pages/project-page.md)は、関連するrunsから記録されたすべてのチャート、データ、モデルが格納される場所です。プロジェクトに名前を付けることで、作業を整理し、1つのプロジェクトに関するすべての情報を1か所に保管できます。

runをプロジェクトに追加するには、`WANDB_PROJECT` 環境変数にプロジェクト名を設定するだけです。`WandbCallback` は、このプロジェクト名の環境変数を取得し、runの設定時に使用します。

<Tabs
  defaultValue="cli"
  values={[
    {label: 'Command Line', value: 'cli'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="cli">

```bash
WANDB_PROJECT=amazon_sentiment_analysis
```

  </TabItem>
  <TabItem value="notebook">

```python
%env WANDB_PROJECT=amazon_sentiment_analysis
```

  </TabItem>
</Tabs>

:::info
`Trainer`を初期化する前に、プロジェクト名を設定してください。
:::

プロジェクト名が指定されていない場合、デフォルトで "huggingface" になります。

### 3) W&Bにトレーニングのrunをログする

これが**最も重要なステップ**です：`Trainer` のトレーニング引数を定義する際、コード内やコマンドラインから、Weights & Biases でのログを有効にするために `report_to` を `"wandb"` に設定します。

また、`run_name`引数を使用してトレーニングrunに名前を付けることもできます。

:::info
TensorFlowを使用していますか？ PyTorchの`Trainer` を TensorFlow の `TFTrainer`に置き換えるだけです。
:::

これで完了です！これで、モデルはトレーニング中に損失、評価メトリクス、モデルトポロジー、勾配をWeights & Biasesにログするようになります。

<Tabs
  defaultValue="cli"
  values={[
    {label: 'Command Line', value: 'cli'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="cli">

```python
python run_glue.py \     # Pythonスクリプトを実行
  --report_to wandb \    # W&Bへのログを有効にする
  --run_name bert-base-high-lr \   # W&Bのrunの名前（オプション）
  # 他のコマンドライン引数はここに
```

</TabItem>
  <TabItem value="notebook">

```python
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    # 他の引数とkwargsはここに
    report_to="wandb",  # W&Bへのログ記録を有効化
    run_name="bert-base-high-lr"  # W&B runの名前（オプション）
)

trainer = Trainer(
    # 他の引数とkwargsはここに
    args=args,  # トレーニング引数
)

trainer.train()  # トレーニングを開始し、W&Bへログを記録
```

  </TabItem>
</Tabs>

#### （ノートブックのみ）W&B Runを終了する

もしトレーニングがPythonスクリプトにカプセル化されている場合、スクリプトが終了するとW&B runも終了します。

JupyterまたはGoogle Colabノートブックを使用している場合は、トレーニングが終了したことを`wandb.finish()`を呼び出すことで伝える必要があります。

```python
trainer.train()  # W&Bにトレーニングとログ記録を開始する
# トレーニング後の分析、テスト、その他のログコード

wandb.finish()
```

### 4) 結果を可視化する

トレーニング結果をログに記録したら、[W&Bダッシュボード](../track/app.md)で結果を動的に探索できます。複数のrunを一度に比較しやすく、興味深い発見をズームインし、柔軟でインタラクティブな可視化で複雑なデータから洞察を引き出すことが容易です。

## おすすめ記事

以下は、TransformersとW&Bに関連するおすすめの6つの記事です。

<details>

<summary>Hugging Face Transformersのハイパーパラメータ最適化</summary>

* Hugging Face Transformersのハイパーパラメータ最適化の3つの戦略 - グリッドサーチ、ベイズ最適化、および集団的学習 - を比較します。
* Hugging Face transformersの標準的なuncased BERTモデルを使用し、SuperGLUEベンチマークのRTEデータセットで微調整を行いたい。
* 結果は、集団的学習がHugging Faceトランスフォーマーモデルのハイパーパラメータ最適化に最も効果的なアプローチであることを示します。

詳細レポートは[こちら](https://wandb.ai/amogkam/transformers/reports/Hyperparameter-Optimization-for-Hugging-Face-Transformers--VmlldzoyMTc2ODI)。
</details>

<details>

<summary>ハグするツイート：ツイートを生成するモデルのトレーニング</summary>

* この記事では、著者が5分で誰のツイートでもHuggingFace Transformerの学習済みGPT2モデルを微調整する方法を実演しています。
* モデルは以下の開発フローを使用しています：ツイートのダウンロード、データセットの最適化、初期実験、ユーザー間の損失の比較、モデルの微調整。

詳細なレポートは、[こちら](https://wandb.ai/wandb/huggingtweets/reports/HuggingTweets-Train-a-Model-to-Generate-Tweets--VmlldzoxMTY5MjI)をご覧ください。
</details>

<details>

<summary>Hugging Face BERTとWBを用いた文章の分類</summary>

* この記事では、自然言語処理（NLP）で最近の画期的な進歩を活用した文章の分類器を作成し、NLPへの転移学習の適用に焦点を当てます。
* Corpus of Linguistic Acceptability（CoLA）データセットを使用して、単一の文章の分類を行います。これは、文法的に正しいか正しくないかというラベルが付けられた文章のセットで、2018年5月に初めて公開されました。
* GoogleのBERTを使用して、NLPタスクの範囲で最小限の努力で高性能なモデルを作成します。

詳細なレポートは[こちら](https://wandb.ai/cayush/bert-finetuning/reports/Sentence-Classification-With-Huggingface-BERT-and-W-B--Vmlldzo4MDMwNA)でご覧いただけます。
</details>

<details>

<summary>Hugging Faceモデルパフォーマンスのトラッキング方法</summary>

* Weights & BiasesとHugging Faceトランスフォーマーを使用して、DistilBERTをトレーニングします。DistilBERTは、BERTより40%小さく、BERTの精度の97%を保持したトランスフォーマーです。
* GLUEベンチマークは、NLPモデルのトレーニング用の9つのデータセットとタスクを集めたものです。

詳細なレポートは[こちら](https://wandb.ai/jxmorris12/huggingface-demo/reports/A-Step-by-Step-Guide-to-Tracking-HuggingFace-Model-Performance--VmlldzoxMDE2MTU)でご覧いただけます。
</details>

<details>

<summary>HuggingFaceでのEarly Stoppingの例</summary>

* Hugging Face TransformerをEarly Stopping正則化を使用してファインチューニングすることは、PyTorchまたはTensorFlowではネイティブに実行できます。
* TensorFlowでは、`tf.keras.callbacks.EarlyStopping`コールバックを使って手軽にEarlyStoppingコールバックを使用できます。
* PyTorchでは、すぐに使えるearly stopping方法はありませんが、GitHub Gistで利用可能なearly stoppingフックがあります。

完全なレポートは[こちら](https://wandb.ai/ayush-thakur/huggingface/reports/Early-Stopping-in-HuggingFace-Examples--Vmlldzo0MzE2MTM)で読めます。
</details>

<details>

<summary>カスタムデータセットでの Hugging Face Transformer の微調整方法</summary>

カスタムIMDBデータセットでセンチメント分析（二値分類）用にDistilBERTトランスフォーマーを微調整します。

完全なレポートは[こちら](https://wandb.ai/ayush-thakur/huggingface/reports/How-to-Fine-Tune-HuggingFace-Transformers-on-a-Custom-Dataset--Vmlldzo0MzQ2MDc)で読めます。
</details>

## 高度な機能

### モデルバージョン管理を有効にする

[Weights & Biases のアーティファクト](https://docs.wandb.ai/artifacts)を使用すると、最大100GBのモデルやデータセットを格納できます。Hugging Face モデルを W&B アーティファクトにログするには、`WANDB_LOG_MODEL` という名前の W&B 環境変数を `'end'` または `'checkpoint'` のいずれかに設定します。
`'end'` は最終モデルのみをログし、`'checkpoint'` は [`save_steps`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.save_steps) ごとにモデルのチェックポイントをログします（[`TrainingArguments`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments)内）。

<Tabs
  defaultValue="cli"
  values={[
    {label: 'コマンドライン', value: 'cli'},
    {label: 'ノートブック', value: 'notebook'},
  ]}>
  <TabItem value="cli">

```bash
WANDB_LOG_MODEL='end'
```
</TabItem>
  <TabItem value="notebook">

```python
%env WANDB_LOG_MODEL='end'
```

  </TabItem>
</Tabs>


:::info
デフォルトでは、`WANDB_LOG_MODEL`が`end`に設定されている場合、モデルはW&B Artifactsに`model-{run_id}`として保存されます。`WANDB_LOG_MODEL`が`checkpoint`に設定されている場合は、`checkpoint-{run_id}`として保存されます。
ただし、`TrainingArguments`で[`run_name`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments.run_name)を渡すと、モデルは`model-{run_name}`または`checkpoint-{run_name}`として保存されます。
:::

これで、これから初期化するすべての`Trainer`はモデルをW&Bプロジェクトにアップロードします。モデルファイルはW&B Artifacts UIで表示できます。モデルとデータセットのバージョン管理に関しては、[Weights & Biases' Artifactsガイド](https://docs.wandb.ai/artifacts)を参照してください。

#### 最良のモデルをどのように保存しますか？

`load_best_model_at_end=True`が`Trainer`に渡された場合、W&Bは最もパフォーマンスの良いモデルをArtifactsに保存します。

### 保存されたモデルの読み込み

`WANDB_LOG_MODEL`でW&B Artifactsにモデルを保存した場合は、追加のトレーニングや推論の実行のためにモデルの重みをダウンロードできます。以前使用したHugging Faceアーキテクチャーに戻して読み込むだけです。

```python
# 新しいrunを作成します
with wandb.init(project="amazon_sentiment_analysis") as run:
# runにアーティファクトを接続する
  my_model_name = "model-bert-base-high-lr:latest"
  my_model_artifact = run.use_artifact(my_model_name)

  # モデルの重みをフォルダにダウンロードし、パスを返す
  model_dir = my_model_artifact.download()

  # そのフォルダからHugging Faceトランスフォーマーを読み込む
  #  同じモデルクラスを使用する
  model = AutoModelForSequenceClassification.from_pretrained(
      model_dir, num_labels=num_labels)

  # 追加のトレーニングを行うか、推論を実行する
```
### チェックポイントからのトレーニング再開
`WANDB_LOG_MODEL='checkpoint'` を設定していた場合、`model_dir` を `TrainingArguments` の `model_name_or_path` 引数に使用し、`resume_from_checkpoint=True` を `Trainer` に渡すことで、トレーニングを再開することもできます。

```python
last_run_id = "xxxxxxxx"  # wandbワークスペースからrun_idを取得

# run_idからwandb runを再開
with wandb.init(
    project=os.environ["WANDB_PROJECT"],
    id=last_run_id,
    resume="must",) as run:
    
  # runにアーティファクトを接続する
  my_checkpoint_name = f"checkpoint-{last_run_id}:latest"
  my_checkpoint_artifact = run.use_artifact(my_model_name)

# フォルダにチェックポイントをダウンロードし、パスを返す
  checkpoint_dir = my_checkpoint_artifact.download()
  
  ＃モデルとトレーナーを再初期化
  model = AutoModelForSequenceClassification.from_pretrained(
      <model_name>, num_labels=num_labels)
  
  ＃素晴らしいトレーニング引数がここにあります。
  training_args = TrainingArguments(...) 
  
  trainer = Trainer(
      model=model,
      args=training_args,
      ...)
  
  ＃チェックポイントからトレーニングを再開するため、チェックポイントディレクトリを使用してください
  trainer.train(resume_from_checkpoint=checkpoint_dir) 
```

### 追加のW&B設定

`Trainer`でログした内容をさらに設定することができます。[ここ](https://docs.wandb.ai/library/environment-variables)でW&B環境変数の全リストを見ることができます。

| 環境変数 | 使用法 |
| ---------- |------------|
| `WANDB_PROJECT` | プロジェクトに名前を付ける（デフォルトでは`huggingface`） |
| `WANDB_LOG_MODEL` | トレーニング終了時にモデルをアーティファクトとしてログする（デフォルトでは`false`） |
| `WANDB_WATCH` | <p>モデルの勾配、パラメータ、またはどちらもログしたいかどうか設定します。</p><ul><li><code>false</code>（デフォルト）：勾配もパラメータもログしない</li><li><code>gradients</code>：勾配のヒストグラムをログする</li><li><code>all</code>：勾配とパラメータのヒストグラムをログする</li></ul> |
| `WANDB_DISABLED` | ログを完全に無効にするには、`true`に設定する（デフォルトは`false`） |
| `WANDB_SILENT` | wandbによって出力される出力を無音にするには、`true`に設定する（デフォルトは`false`） |
<Tabs
  defaultValue="cli"
  values={[
    {label: 'コマンドライン', value: 'cli'},
    {label: 'ノートブック', value: 'notebook'},
  ]}>
  <TabItem value="cli">

```bash
WANDB_WATCH=all
WANDB_SILENT=true
```

  </TabItem>
  <TabItem value="notebook">

```python
%env WANDB_WATCH=all
%env WANDB_SILENT=true
```

  </TabItem>
</Tabs>

### `wandb.init` をカスタマイズする

`Trainer`が使っている`WandbCallback`は、`Trainer`が初期化されたときに`wandb.init`を呼び出します。代わりに、`Trainer`が初期化される前に`wandb.init`を呼び出して、runを手動で設定することもできます。これにより、W&B run設定を完全にコントロールすることができます。

`init`に渡す例を下記に示します。`wandb.init`の使い方の詳細は、[リファレンスドキュメントをご覧ください](../../ref/python/init.md) 。
```python
wandb.init(project="amazon_sentiment_analysis",
           name="bert-base-high-lr",
           tags=["baseline", "high-lr"],
           group="bert")
```

### カスタムログ

Weights & Biasesへのログは、[Transformersの`Trainer`](https://huggingface.co/transformers/main_classes/trainer.html) によって、Transformersライブラリ内の`WandbCallback`([参考資料](https://huggingface.co/transformers/main_classes/callback.html#transformers.integrations.WandbCallback))で処理されます。Hugging Faceのログをカスタマイズする必要がある場合は、このコールバックを変更してください。

## 問題点、質問、機能のリクエスト

Hugging FaceのW&Bインテグレーションに関する問題、質問、機能のリクエストは、[Hugging Faceフォーラムのこのスレッド](https://discuss.huggingface.co/t/logging-experiment-tracking-with-w-b/498)で投稿するか、Hugging Face [Transformers GitHubリポジトリ](https://github.com/huggingface/transformers)でissueを立ててください。