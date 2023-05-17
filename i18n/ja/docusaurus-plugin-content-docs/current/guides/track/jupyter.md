---
description: Jupyterを使ってWeights & Biasesを使用し、ノートブックを離れることなくインタラクティブな可視化を取得しましょう。
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Jupyterノートブックのトラッキング

<head>
  <title>Jupyterノートブックのトラッキング</title>
</head>

Jupyterを使ってWeights & Biasesを使用し、ノートブックを離れることなくインタラクティブな可視化を取得しましょう。カスタム分析、実験、プロトタイプを組み合わせて、すべて完全に記録されます！

## W&BとJupyterノートブックのユースケース

1. **繰り返し実験**: 実験を実行および再実行し、パラメーターを調整し、途中で手動のメモを取らずにすべてのrunをW&Bに自動的に保存します。
2. **コード保存**: モデルを再現するとき、ノートブックのどのセルが実行され、どの順序で実行されたかわかりにくいです。[設定ページ](../app/settings-page/intro.md) でコード保存をオンにして、各実験のセル実行の記録を保存しましょう。
3. **カスタム分析**: runがW&Bに記録されると、APIからデータフレームを取得し、カスタム分析を行い、その結果をW&Bに記録してレポートで保存および共有するのが簡単になります。

## ノートブックでの始め方

W&Bをインストールし、アカウントをリンクするために以下のコードでノートブックを開始します。

```python
!pip install wandb -qqq
import wandb
wandb.login()
```

次に、実験を設定し、ハイパーパラメーターを保存します:

```python
wandb.init(project="jupyter-projo",
           config={
               "batch_size": 128,
               "learning_rate": 0.01,
               "dataset": "CIFAR-100",
           })
```

`wandb.init()` を実行した後、`%%wandb`で新しいセルを開始して、ノートブック内でリアルタイムのグラフを表示します。このセルを複数回実行すると、データがrunに追加されます。

```python
%%wandb

# ここにトレーニングループを記述
```

[クイック実例ノートブック →](http://wandb.me/jupyter-interact-colab) で自分自身で試してみましょう。

![](/images/track/jupyter_widget.png)

### ノートブック内でのリアルタイムのW&Bインターフェースのレンダリング

`%wandb` マジックを使用して、既存のダッシュボード、スイープ、レポートをノートブック内に直接表示することもできます。

```python
# プロジェクトワークスペースを表示
%wandb USERNAME/PROJECT
# シングルのrunを表示
%wandb USERNAME/PROJECT/runs/RUN_ID
# スイープの表示
%wandb ユーザー名/プロジェクト/sweeps/SWEEP_ID
# レポートの表示
%wandb ユーザー名/プロジェクト/reports/REPORT_ID
# 埋め込まれたiframeの高さを指定
%wandb ユーザー名/プロジェクト -h 2048
```

`%%wandb`や`%wandb`マジックとして代替手段として、`wandb.init()`を実行した後、任意のセルで`wandb.run`を使用してインライングラフを表示することができます。また、APIから取得したレポート、スイープ、runオブジェクトに`ipython.display(...)`を呼び出すことができます。

```python
# 最初にwandb.runを初期化
wandb.init()

# セルの出力がwandb.runの場合、ライブグラフが表示されます
wandb.run
```

:::info
W&Bでできることの詳細は、[データとメディアの記録に関するガイド](log/intro.md)を参照してください。お気に入りのMLツールキットとの統合方法について学ぶには、[こちら](../integrations/intro.md)を参照してください。また、[リファレンスドキュメント](../../ref/python/README.md)や[例のリポジトリ](https://github.com/wandb/examples)を直接閲覧してください。
:::

## W&Bでの追加のJupyter機能

1. **Colabでの簡単な認証**：Colabで初めて`wandb.init`を呼び出すと、ブラウザでW&Bにログインしている場合、自動的にランタイムが認証されます。ランの概要タブにはColabへのリンクが表示されます。
2. **Jupyter Magic**：ダッシュボード、スイープ、レポートをノートブックに直接表示します。`%wandb`マジックは、プロジェクト、スイープ、レポートへのパスを受け取り、ノートブックに直接W&Bインターフェースを表示します。
3. **Docker化されたJupyterを起動**：`wandb docker --jupyter`を実行してdockerコンテナを起動し、コードをマウントし、Jupyterをインストールしてポート8888で起動します。
4. **任意の順序でセルを実行することを恐れない**：デフォルトでは、次に`wandb.init`が呼び出されるまでrunを"終了"としてマークしません。これにより、複数のセル（データの設定、トレーニング、テストなど）を好きな順番で実行し、すべて同じrunに記録させることができます。[設定](https://app.wandb.ai/settings)でコード保存をオンにすると、実行されたセルの順序と実行状態が記録され、非線形な開発フローでも再現が可能になります。Jupyterノートブックでrunを手動で終了するには、`run.finish`を呼び出してください。

```python
import wandb
run = wandb.init()

# ここにトレーニングスクリプトとログが入ります

run.finish()
```

## よくある質問

### W&Bの情報メッセージを非表示にする方法は？

標準のwandbログや情報メッセージ（例：プロジェクト情報などの開始時の情報）を無効にするには、`wandb.login` を実行する前に、ノートブックのセルで以下のコードを実行してください：

<Tabs
  defaultValue="jupyter"
  values={[
    {label: 'Jupyter Magic', value: 'jupyter'},
    {label: 'Python', value: 'python'},
  ]}>
  <TabItem value="jupyter">

```python
%env WANDB_SILENT=True
```
  </TabItem>
  <TabItem value="python">

```python
import os

os.environ["WANDB_SILENT"] = "True"
```
  </TabItem>
</Tabs>

ノートブックに`INFO SenderThread:11484 [sender.py:finish():979]`のようなログメッセージが表示される場合は、以下の方法でそれらを無効にできます。

```python
import logging

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)
```

### `WANDB_NOTEBOOK_NAME`をどのように設定しますか？

`"Failed to query for notebook name, you can set it manually with the WANDB_NOTEBOOK_NAME environment variable,"`というエラーメッセージが表示される場合は、環境変数を設定することで解決できます。方法はいくつかあります：

<Tabs
  defaultValue="jupyter"
  values={[
    {label: 'Jupyter Magic', value: 'jupyter'},
    {label: 'python', value: 'python'},
  ]}>
  <TabItem value="jupyter">

```python
%env "WANDB_NOTEBOOK_NAME" "notebook name here"
```
  </TabItem>
  <TabItem value="python">

```python

import os



os.environ["WANDB_NOTEBOOK_NAME"] = "ここにノートブック名を入力"

```

  </TabItem>

</Tabs>