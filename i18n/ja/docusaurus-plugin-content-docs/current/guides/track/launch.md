---
description: W&B実験を作成する。
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 実験を作成する

<head>
  <title>W&B実験を始める</title>
</head>

W&B Python SDKを使って、機械学習実験をトラッキングします。インタラクティブなダッシュボードで結果を確認するか、[W&B Public API](../../ref/python/public-api/README.md) を使ってPythonにデータをエクスポートしてプログラマティックにアクセスできます。

このガイドでは、W&Bの構成要素を使って W&B実験を作成する方法について説明します。

## W&B実験の作成方法

W&B実験を作成するための４つのステップ：

1. [W&B Runを初期化する](#import-wandb-and-call-wandbinit)
2. [ハイパーパラメータのディクショナリを取得する](#capture-a-dictionary-of-hyperparameters)
3. [トレーニングループ内でメトリクスをログする](#log-metrics-inside-your-training-loop)
4. [アーティファクトをW&Bにログする](#log-an-artifact-to-wb)

### W&B Runを初期化する
スクリプトの最初に、[`wandb.init()`](../../ref/python/init.md) APIを呼び出して、W&B Runとしてデータを同期・ログするバックグラウンドプロセスを生成します。

以下のコードスニペットは、`"cat-classification"`という名前の新しいW&Bプロジェクトを作成する方法を示しています。このrunを識別するために、`"My first experiment"`というノートが追加されています。タグ`"baseline"` と `"paper1"` が含まれており、このrunは将来の論文出版に向けたベースライン実験であることを示しています。

```python
# W&B Pythonライブラリをインポート
import wandb

# 1. W&BのRunを開始
run = wandb.init(
  project="cat-classification",
  notes="My first experiment",
  tags=["baseline", "paper1"]
)
```
`wandb.init()`を使ってW&Bを初期化すると、[Run](../../ref/python/run.md)オブジェクトが返されます。 さらに、W&Bはすべてのログとファイルが保存され、非同期でW&Bサーバーにストリーミングされるローカルディレクトリを作成します。

:::info
注意：wandb.init()を呼び出すと、すでに存在するプロジェクトにRunが追加されます。たとえば、すでに`"cat-classification"`というプロジェクトがある場合、そのプロジェクトは削除されずに継続して存在します。代わりに、新しいRunがプロジェクトに追加されます。
:::

### ハイパーパラメーターの辞書を取得する
学習率やモデルタイプなどのハイパーパラメーターの辞書を保存します。後で実験結果を整理したり問い合わせたりする際に、configで取得したモデル設定が役立ちます。

```python
# 2. ハイパーパラメータの辞書を取得する
wandb.config = {
  "epochs": 100, 
  "learning_rate": 0.001, 
  "batch_size": 128
}
```
実験の設定方法についての詳細は、[実験の設定](./config.md)を参照してください。

### トレーニングループ内でメトリクスを記録する
各`for`ループ（エポック）でメトリクスを記録し、精度と損失値が計算され、[`wandb.log()`](../../ref/python/log.md)を使用してW&Bに記録されます。デフォルトでは、wandb.logを呼び出すと、historyオブジェクトに新しいステップが追加され、summaryオブジェクトが更新されます。
以下のコード例は、`wandb.log`を使用してメトリクスを記録する方法を示しています。

:::note
モデルの設定方法やデータの取得方法に関する詳細は省略されています。
:::

```python
# モデルとデータを設定
model, dataloader = get_model(), get_data()

for epoch in range(wandb.config.epochs):
    for batch in dataloader:
        loss, accuracy = model.training_step()
        # 3. トレーニングループ内でメトリクスを記録し、
        # モデルのパフォーマンスを視覚化する
        wandb.log({"accuracy": accuracy, "loss": loss})
```
W&Bで記録できるさまざまなデータタイプの詳細については、[実験中にデータを記録する](./log/intro.md) を参照してください。

### W&Bにアーティファクトを記録する
オプションでW&B Artifactも記録できます。Artifactsを使用するとデータセットやモデルのバージョン管理が簡単になります。
```python
wandb.log_artifact(model)
```
Artifactsに関する詳細は、[Artifactsの章](../artifacts/intro.md) を参照してください。モデルのバージョン管理に関する詳細は、[モデル管理](../models/intro.md) を参照してください。

### すべてをまとめる
上記のコードスニペットを含む完全なスクリプトは以下にあります：
```python
# W&B Pythonライブラリをインポート
import wandb

# 1. W&BのRunを開始する
run = wandb.init(
    project="cat-classification",
    notes="",
    tags=["baseline", "paper1"]
)

# 2. ハイパーパラメーターの辞書を取得する
wandb.config = {
        "epochs": 100, 
        "learning_rate": 0.001, 
        "batch_size": 128
}

# モデルとデータをセットアップ
model, dataloader = get_model(), get_data()

for epoch in range(wandb.config.epochs):
    for batch in dataloader:
        loss, accuracy = model.training_step()
    # 3. トレーニングループ内でメトリクスを記録し、モデルの性能を可視化する
        wandb.log({"accuracy": accuracy, "loss": loss})

# 4. アーティファクトをW&Bにログする
wandb.log_artifact(model)
# オプション: 最後にモデルを保存
model.to_onnx()
wandb.save("model.onnx")
```

## 次のステップ: 実験を可視化する
W&Bダッシュボードを、機械学習モデルの結果を整理・可視化するための中心的な場所として利用します。数回クリックするだけで、[並行座標プロット](../app/features/panels/parallel-coordinates.md)、[パラメータ重要度解析](../app/features/panels/parameter-importance.md)、[その他](../app/features/panels/intro.md) のような豊富でインタラクティブなチャートを作成することができます。

![Quickstartスイープダッシュボードの例](/images/sweeps/quickstart_dashboard_example.png)

実験や特定のrunsの表示方法については、[実験の結果を可視化する](./app.md)を参照してください。


## ベストプラクティス
以下は、実験を作成する際に考慮すべきいくつかのガイドラインを提案します。

1. **設定( Config)**: ハイパーパラメーター、アーキテクチャ、データセットなど、モデルの再現に使用したい情報をトラッキングします。これらは列に表示され、アプリで動的にrunsをグループ化、並べ替え、フィルターできます。
2. **プロジェクト**: プロジェクトは、一緒に比較できる一連の実験です。各プロジェクトには専用のダッシュボードページが用意されており、異なるモデルのバージョンを比較するために様々なrunsのグループを簡単に有効・無効にできます。
3. **メモ**: 自分への簡単なコミットメッセージです。メモはスクリプトから設定でき、後でW&Bアプリのプロジェクトダッシュボードの概要セクションで編集できます。
4. **タグ**: ベースラインのrunsやお気に入りのrunsを識別します。タグを使用してrunsをフィルタリングできます。タグは後でW&Bアプリのプロジェクトダッシュボードの概要セクションで編集できます。

以下のコードスニペットは、上記のベストプラクティスを使用してW&B実験を定義する方法を示しています。

```python
import wandb

config = dict (
  learning_rate = 0.01,
  momentum = 0.2,
  architecture = "CNN",
  dataset_id = "cats-0192"
)
wandb.init(

  project="detect-cats",

  notes="ベースラインの調整",

  tags=["ベースライン", "paper1"],

  config=config,

)

```



W&B実験を定義する際の利用可能なパラメータについての詳細は、[APIリファレンスガイド](../../ref/python/README.md)の[`wandb.init`](../../ref/python/init.md) APIドキュメントを参照してください。