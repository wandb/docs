---
description: Jupyterと一緒にW&Bを使用して、ノートブックを離れることなくインタラクティブな可視化を取得します。
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Jupyterノートブックのトラッキング

<head>
  <title>Jupyterノートブックのトラッキング</title>
</head>

W&BをJupyterと一緒に使うことで、ノートブックを離れることなくインタラクティブな可視化ができます。カスタム分析、Experiments、プロトタイプを組み合わせて、すべてを完全にログします！

## JupyterノートブックとW&Bのユースケース

1. **反復的な実験**: パラメータを調整しながら実験を何度も実行し、それらのRunsは手動でメモを取らなくても自動的にW&Bに保存されます。
2. **コードの保存**: モデルを再現する際、ノートブックのどのセルが実行され、どの順序で実行されたかを知るのは難しいです。[設定ページ](../app/settings-page/intro.md)でコード保存をオンにして、各Experimentのセル実行の記録を保存しましょう。
3. **カスタム分析**: RunsがW&Bにログされると、APIからデータフレームを取得しカスタム分析を行い、それらの結果をW&Bにログしてレポートで保存および共有できます。

## ノートブックでの始め方

以下のコードでノートブックを開始し、W&Bをインストールしてアカウントにリンクします:

```notebook
!pip install wandb -qqq
import wandb
wandb.login()
```

次に、Experimentを設定しハイパーパラメータを保存します:

```python
wandb.init(
    project="jupyter-projo",
    config={
        "batch_size": 128,
        "learning_rate": 0.01,
        "dataset": "CIFAR-100",
    },
)
```

`wandb.init()` を実行した後、新しいセルを `%%wandb` で開始すると、ノートブック内でライブグラフを表示できます。このセルを複数回実行すると、データがそのRunに追加されます。

```notebook
%%wandb

# トレーニングループをここに
```

この[クイック例ノートブック](http://wandb.me/jupyter-interact-colab)で自分で試してみてください。

![](/images/track/jupyter_widget.png)

### ノートブック内でライブW&Bインターフェースを直接レンダリング

任意のダッシュボード、sweeps、またはレポートをノートブック内で直接表示することもできます。`%wandb`マジックを使用します:

```notebook
# プロジェクトのワークスペースを表示
%wandb USERNAME/PROJECT
# 単一のrunを表示
%wandb USERNAME/PROJECT/runs/RUN_ID
# sweepを表示
%wandb USERNAME/PROJECT/sweeps/SWEEP_ID
# レポートを表示
%wandb USERNAME/PROJECT/reports/REPORT_ID
# 埋め込みiframeの高さを指定
%wandb USERNAME/PROJECT -h 2048
```

`%%wandb` や `%wandb` マジックの代替として、`wandb.init()` を実行した後に任意のセルを `wandb.run` で終えるとインライングラフが表示されます。また、apiから返される任意のレポート、sweep、またはrunオブジェクトに対して `ipython.display(...)` を呼び出すこともできます。

```python
# まずはwandb.runを初期化
wandb.init()

# セルの出力がwandb.runなら、ライブグラフが表示されます
wandb.run
```

:::情報
W&Bでできることについてもっと知りたいですか？ 私たちの[データとメディアのログガイド](log/intro.md)をチェックするか、お気に入りのMLツールキットとW&Bを[統合する方法](../integrations/intro.md)を学び、または[リファレンスドキュメント](../../ref/python/README.md)や[サンプルのリポジトリ](https://github.com/wandb/examples)を直接ご覧ください。
:::

## W&Bでの追加のJupyter機能

1. **Colabでの簡単な認証**: Colabで初めて `wandb.init` を呼び出すとき、ブラウザでW&Bにログインしている場合、ランタイムが自動的に認証されます。runページのOverviewタブにはColabへのリンクが表示されます。
2. **Jupyter Magic**: ダッシュボード、sweeps、レポートをノートブック内で直接表示します。`%wandb` マジックはプロジェクト、sweeps、またはレポートへのパスを受け取り、W&Bインターフェースをノートブック内で直接レンダリングします。
3. **Docker化されたJupyterの起動**: `wandb docker --jupyter` を呼び出してdockerコンテナを起動し、コードをマウントし、Jupyterがインストールされていることを確認し、ポート8888で起動します。
4. **任意の順序で安全にセルを実行**: デフォルトでは、次に `wandb.init` が呼び出されるまでrunを「完了」としてマークしません。それにより、複数のセル（例えば、データの設定、トレーニング、テストの各セル）を好きな順序で実行し、すべてを同じrunにログすることができます。[設定](https://app.wandb.ai/settings)でコード保存をオンにすると、実行されたセルを順序通りおよび実行された状態でログし、最も非線形なパイプラインでも再現可能にします。Jupyterノートブックでrunを手動で完了としてマークするには、`run.finish` を呼び出します。

```python
import wandb

run = wandb.init()

# トレーニングスクリプトとログがここに入ります

run.finish()
```

## よくある質問

### W&Bの情報メッセージを消すにはどうすればいいですか？

標準のwandbログや情報メッセージ（例：runの開始時のプロジェクト情報）を無効にするには、`wandb.login` を実行する _前に_ 次のコードをノートブックのセルに入力します:

<Tabs
  defaultValue="jupyter"
  values={[
    {label: 'Jupyter Magic', value: 'jupyter'},
    {label: 'Python', value: 'python'},
  ]}>
  <TabItem value="jupyter">

```notebook
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

ノートブックに `INFO SenderThread:11484 [sender.py:finish():979]` のようなログメッセージが表示される場合は、次のコードでそれらを無効にできます:

```python
import logging

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)
```

### `WANDB_NOTEBOOK_NAME` を設定するにはどうすればいいですか？

エラーメッセージ `"Failed to query for notebook name, you can set it manually with the WANDB_NOTEBOOK_NAME environment variable,"` が表示される場合、環境変数を設定して解決できます。設定する方法はいくつかあります:

<Tabs
  defaultValue="jupyter"
  values={[
    {label: 'Jupyter Magic', value: 'jupyter'},
    {label: 'python', value: 'python'},
  ]}>
  <TabItem value="jupyter">

```notebook
%env "WANDB_NOTEBOOK_NAME" "notebook name here"
```
  </TabItem>
  <TabItem value="python">

```notebook
import os

os.environ["WANDB_NOTEBOOK_NAME"] = "notebook name here"
```
  </TabItem>
</Tabs>