---
title: Track Jupyter notebooks
description: Jupyter で W&B を使用して、ノートブック を離れることなくインタラクティブな 可視化 を取得します。
menu:
  default:
    identifier: ja-guides-models-track-jupyter
    parent: experiments
weight: 6
---

Jupyter を W&B と共に使用すると、ノートブックから離れることなくインタラクティブな可視化ができます。カスタム 分析 、 実験 、プロトタイプを組み合わせて、すべて完全にログに記録されます。

## Jupyter ノートブックと W&B の ユースケース

1.  **反復的な 実験 **: パラメータを微調整して 実験 を実行および再実行すると、実行したすべての run が自動的に W&B に保存されるため、途中で手動でメモを取る必要がありません。
2.  **コード の保存**: モデル を再現する場合、ノートブックのどのセルがどの順序で実行されたかを知ることは困難です。[ 設定 ページ]({{< relref path="/guides/models/app/settings-page/" lang="ja" >}}) でコード の保存をオンにすると、各 実験 のセル実行の記録を保存できます。
3.  **カスタム 分析 **: run が W&B に記録されると、API からデータフレームを取得してカスタム 分析 を行い、それらの 結果 を W&B に記録して レポート で保存および共有することが簡単になります。

## ノートブックで始める

次の コード でノートブックを開始して、W&B をインストールし、アカウントをリンクします。

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

`wandb.init()` を実行した後、`%%wandb` で新しいセルを開始して、ノートブックでライブグラフを表示します。このセルを複数回実行すると、データが run に追加されます。

```notebook
%%wandb

# Your training loop here
```

この[ノートブック の例](http://wandb.me/jupyter-interact-colab)でお試しください。

{{< img src="/images/track/jupyter_widget.png" alt="" >}}

### ノートブックでライブ W&B インターフェイスを直接レンダリングする

`%wandb` マジックを使用して、既存の ダッシュボード 、 Sweeps 、または Reports をノートブックで直接表示することもできます。

```notebook
# Display a project workspace
%wandb USERNAME/PROJECT
# Display a single run
%wandb USERNAME/PROJECT/runs/RUN_ID
# Display a sweep
%wandb USERNAME/PROJECT/sweeps/SWEEP_ID
# Display a report
%wandb USERNAME/PROJECT/reports/REPORT_ID
# Specify the height of embedded iframe
%wandb USERNAME/PROJECT -h 2048
```

`%%wandb` または `%wandb` マジックの代わりに、`wandb.init()` を実行した後、セルを `wandb.run` で終了してインライングラフを表示するか、API から返された report 、 sweep 、または run オブジェクトで `ipython.display(...)` を呼び出すことができます。

```python
# Initialize wandb.run first
wandb.init()

# If cell outputs wandb.run, you'll see live graphs
wandb.run
```

{{% alert %}}
W&B で何ができるかについてもっと知りたいですか？[データとメディアをログに記録する ガイド ]({{< relref path="/guides/models/track/log/" lang="ja" >}}) を確認するか、[お気に入りの ML ツールキットと 統合 する方法]({{< relref path="/guides/integrations/" lang="ja" >}}) を学ぶか、[リファレンス ドキュメント]({{< relref path="/ref/python/" lang="ja" >}}) または[examples の repo](https://github.com/wandb/examples)に直接飛び込んでください。
{{% /alert %}}

## W&B の追加の Jupyter 機能

1.  **Colab での簡単な認証**: Colab で `wandb.init` を初めて呼び出すと、ブラウザで W&B に現在ログインしている場合、ランタイムが自動的に認証されます。run ページの Overviewタブ に、Colab へのリンクが表示されます。
2.  **Jupyter Magic:** ダッシュボード 、 Sweeps 、 Reports をノートブックで直接表示します。`%wandb` マジックは、 project 、 Sweeps 、または Reports へのパスを受け入れ、W&B インターフェイスをノートブックで直接レンダリングします。
3.  **docker container 化された Jupyter を Launch**: `wandb docker --jupyter` を呼び出して docker container を Launch し、 コード をマウントし、Jupyter がインストールされていることを確認して、ポート 8888 で Launch します。
4.  **恐れることなく任意の順序でセルを実行**: デフォルトでは、run を `finished` としてマークするために、次回の `wandb.init` が呼び出されるまで待機します。これにより、複数のセル (たとえば、データを設定するセル、トレーニングするセル、テストするセル) を好きな順序で実行し、それらすべてを同じ run に記録できます。[ 設定 ](https://app.wandb.ai/settings) で コード の保存をオンにすると、実行されたセルも、実行された順序と実行された状態で記録され、最も非線形の パイプライン でさえ再現できるようになります。Jupyter ノートブックで run を手動で完了としてマークするには、`run.finish` を呼び出します。

```python
import wandb

run = wandb.init()

# training script and logging goes here

run.finish()
```
