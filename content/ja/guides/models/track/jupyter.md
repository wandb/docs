---
title: Track Jupyter notebooks
description: Jupyter で W&B を使用して、 ノートブック から離れることなくインタラクティブな 可視化 を取得できます。
menu:
  default:
    identifier: ja-guides-models-track-jupyter
    parent: experiments
weight: 6
---

Jupyter で W&B を使用すると、ノートブックから離れることなくインタラクティブな可視化を得ることができます。カスタムの 分析 、 実験 、プロトタイプを組み合わせて、すべて完全に ログ に記録します。

## Jupyter ノートブック で W&B を使用する ユースケース

1. **反復的な 実験 **: パラメータを調整しながら 実験 を実行および再実行すると、手動でメモを取らなくても、実行したすべての run が W&B に自動的に保存されます。
2. **コード の保存**: モデル を再現する際、ノートブック のどのセルがどの順序で実行されたかを知ることは困難です。[ 設定 ページ]({{< relref path="/guides/models/app/settings-page/" lang="ja" >}}) で コード の保存をオンにすると、各 実験 のセル実行の記録が保存されます。
3. **カスタム 分析 **: run が W&B に ログ されると、API からデータフレームを取得してカスタム 分析 を実行し、それらの 結果 を W&B に ログ して レポート で保存および共有することが簡単になります。

## ノートブック での始め方

次の コード でノートブックを開始して W&B をインストールし、アカウントをリンクします。

```notebook
!pip install wandb -qqq
import wandb
wandb.login()
```

次に、 実験 を設定し、 ハイパーパラメータ を保存します。

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

`wandb.init()` を実行した後、`%%wandb` を使用して新しいセルを開始すると、ノートブック にライブグラフが表示されます。このセルを複数回実行すると、データが run に追加されます。

```notebook
%%wandb

# Your training loop here
# ここにトレーニングループを記述します
```

この[サンプルノートブック](http://wandb.me/jupyter-interact-colab)でお試しください。

{{< img src="/images/track/jupyter_widget.png" alt="" >}}

### ノートブック でライブ W&B インターフェースを直接レンダリングする

`%wandb` マジックを使用して、既存の ダッシュボード 、 Sweeps 、または Reports をノートブック で直接表示することもできます。

```notebook
# Display a project workspace
# プロジェクトワークスペースを表示する
%wandb USERNAME/PROJECT
# Display a single run
# 単一のrunを表示する
%wandb USERNAME/PROJECT/runs/RUN_ID
# Display a sweep
# sweepを表示する
%wandb USERNAME/PROJECT/sweeps/SWEEP_ID
# Display a report
# reportを表示する
%wandb USERNAME/PROJECT/reports/REPORT_ID
# Specify the height of embedded iframe
# 埋め込みiframeの高さを指定する
%wandb USERNAME/PROJECT -h 2048
```

`%%wandb` または `%wandb` マジックの代わりに、`wandb.init()` を実行した後、`wandb.run` で任意のセルを終了してインライングラフを表示するか、API から返された report 、 sweep 、または run オブジェクトで `ipython.display(...)` を呼び出すことができます。

```python
# Initialize wandb.run first
# 最初にwandb.runを初期化します
wandb.init()

# If cell outputs wandb.run, you'll see live graphs
# セルがwandb.runを出力する場合、ライブグラフが表示されます
wandb.run
```

{{% alert %}}
W&B で何ができるかについてもっと知りたいですか？[データとメディアの ログ に関する ガイド]({{< relref path="/guides/models/track/log/" lang="ja" >}}) を確認するか、[お気に入りの ML ツールキットとの 統合 方法]({{< relref path="/guides/integrations/" lang="ja" >}}) を学ぶか、[リファレンス ドキュメント]({{< relref path="/ref/python/" lang="ja" >}}) または[ 例 の レポジトリ](https://github.com/wandb/examples)に直接飛び込んでください。
{{% /alert %}}

## W&B の追加の Jupyter 機能

1. **Colab での簡単な認証**: Colab で `wandb.init` を初めて呼び出すと、ブラウザで W&B に現在 ログイン している場合、 ランタイム が自動的に認証されます。run ページの Overviewタブ に、Colab へのリンクが表示されます。
2. **Jupyter Magic:** ダッシュボード 、 Sweeps 、および Reports をノートブック で直接表示します。`%wandb` マジックは、 プロジェクト 、 Sweeps 、または Reports へのパスを受け入れ、W&B インターフェースをノートブック で直接レンダリングします。
3. **docker コンテナ 化された Jupyter の Launch**: `wandb docker --jupyter` を呼び出して dockerコンテナ を起動し、 コード をマウントし、Jupyter がインストールされていることを確認して、ポート 8888 で起動します。
4. **恐れることなく任意の順序でセルを実行する**: デフォルトでは、run を `finished` としてマークするために、次に `wandb.init` が呼び出されるまで待機します。これにより、複数のセル（たとえば、データをセットアップするセル、 トレーニング するセル、テストするセル）を好きな順序で実行し、すべて同じ run に ログ できます。[ 設定 ](https://app.wandb.ai/settings)で コード の保存をオンにすると、実行されたセルも、実行された順序と実行された状態で ログ に記録され、最も非線形の パイプライン でも再現できます。Jupyter ノートブック で run を手動で完了としてマークするには、`run.finish` を呼び出します。

```python
import wandb

run = wandb.init()

# training script and logging goes here
# ここにトレーニングスクリプトとロギングを記述します

run.finish()
```