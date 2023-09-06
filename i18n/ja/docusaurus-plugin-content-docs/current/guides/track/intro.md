---
description: Track machine learning experiments with W&B.
slug: /guides/track
displayed_sidebar: ja
---


# 実験のトラッキング

<head>
  <title>機械学習とディープラーニングの実験をトラックする</title>
</head>

W&BのPythonライブラリを使って、数行のコードで機械学習の実験をトラッキングできます。その後、[インタラクティブなダッシュボード](app.md)で結果を確認したり、[Public API](../../ref/python/public-api/README.md)を使ってPythonにデータをエクスポートしてプログラムでアクセスできます。

[PyTorch](../integrations/pytorch.md)、[Keras](../integrations/keras.md)、[Scikit](../integrations/scikit.md)などの一般的なフレームワークを使っている場合は、W&Bのインテグレーションを活用してください。[インテグレーションガイド](../integrations/intro.md)で、全てのインテグレーションとW&Bをコードに追加する方法についての情報を入手できます。

## 仕組み

W&Bの実験は、以下の構成要素で構成されています:

1. [**`wandb.init()`**](./launch.md): スクリプトの先頭で新しいrunを初期化します。これにより、`Run`オブジェクトが返され、ログやファイルが保存されるローカルディレクトリが作成され、W&Bサーバーに非同期でストリーミングされます。ホストされたクラウドサーバーの代わりにプライベートサーバーを使用したい場合は、[Self-Hosting](../hosting/intro.md)を提供しています。
2. [**`wandb.config`**](./config.md): 学習率やモデルタイプなどのハイパーパラメータ辞書を保存します。configでキャプチャしたモデルの設定は、後で結果を整理してクエリするときに便利です。
3. [**`wandb.log()`**](./log/intro.md): トレーニングループで精度や損失などのメトリクスを時間と共に記録します。デフォルトでは、`wandb.log`を呼び出すと、`history`オブジェクトに新しいステップが追加され、`summary`オブジェクトが更新されます。
   * `history`: 時間経過と共にメトリクスを記録する辞書のようなオブジェクトの配列。これらの時系列値は、デフォルトでUIの折れ線グラフとして表示されます。
   * `summary`: デフォルトで、wandb.log()で記録されたメトリックの最終値です。メトリックのサマリーを手動で設定して、最終値の代わりに最も高い精度や最も低い損失を記録できます。これらの値は、テーブルやrunを比較するグラフに使用されます。例えば、プロジェクト内のすべてのrunの最終精度を視覚化できます。
4. [**`wandb.log_artifact`**](../../ref/python/artifact.md): モデルの重みや予測のテーブルなど、runの出力を保存します。これにより、モデルのトレーニングだけでなく、最終モデルに影響を与える開発フローの全てのステップを追跡できます。

以下の疑似コードは、一般的なW&B実験のトラッキングワークフローを示しています:

```python
# 任意のPythonスクリプトに対する柔軟なインテグレーション
import wandb

# 1. W&B Runを開始する
wandb.init(project="my-project-name")

# 2. モード入力とハイパーパラメーターを保存する
config = wandb.config
config.learning_rate = 0.01

# モデルとデータを設定する
model, dataloader = get_model(), get_data()

# モデルトレーニングはここに入ります

# 3. メトリクスを時系列でログすることでパフォーマンスを可視化する
wandb.log({"loss": loss})

# 4. W&Bにアーティファクトをログする
wandb.log_artifact(model)
```

## はじめ方

あなたのユースケースに応じて、以下のリソースを参考にしてW&B実験を始めてください：

* W&B実験を初めて利用する場合は、Quick Startをお読みください。[クイックスタート](../../quickstart.md)では、初めての実験を設定する方法について説明しています。
* W&B Developer Guideの「実験」に関するトピックを探索してください：
  * 実験を作成する
  * 実験を設定する
  * 実験からのデータをログする
  * 実験からの結果を表示する
* [W&B API Reference Guide](../../ref/README.md)内にある[W&B Pythonライブラリ](../../ref/python/README.md)を参照してください。