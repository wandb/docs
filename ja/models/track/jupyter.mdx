---
title: Jupyter ノートブックをトラッキングする
description: Jupyter を使用して W&B と連携し、ノートブックから離れることなくインタラクティブな可視化を得ましょう。
menu:
  default:
    identifier: ja-guides-models-track-jupyter
    parent: experiments
weight: 6
---

W&B を Jupyter と組み合わせることで、ノートブックを離れることなくインタラクティブな可視化を実現できます。カスタム分析、実験管理、プロトタイプをすべて完全にログしながら結合します。

## Jupyter ノートブックにおける W&B のユースケース

1. **反復実験**: 実験を実行および再実行して、パラメータを調整し、すべての実行を手動でメモを取ることなく自動的に W&B に保存します。
2. **コード保存**: モデルを再現する際、ノートブックのどのセルが実行されたか、どの順序で実行されたかを知るのは難しいです。各実験のセル実行の記録を保存するために、[設定ページ]({{< relref path="/guides/models/app/settings-page/" lang="ja" >}})でコード保存をオンにしてください。
3. **カスタム分析**: 実行が W&B にログされたら、APIからデータフレームを取得して、カスタム分析を行い、その結果を W&B にログしてレポートで保存し、共有できます。

## ノートブックでの始め方

W&B をインストールしてアカウントをリンクするために、次のコードでノートブックを開始します：

```notebook
!pip install wandb -qqq
import wandb
wandb.login()
```

次に、実験を設定してハイパーパラメーターを保存します：

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

`wandb.init()` を実行した後、新しいセルを `%%wandb` から開始して、ノートブックでライブグラフを表示します。このセルを複数回実行すると、データは run に追加されます。

```notebook
%%wandb

# ここにトレーニングループ
```

この[例のノートブック](http://wandb.me/jupyter-interact-colab)で試してみてください。

{{< img src="/images/track/jupyter_widget.png" alt="" >}}

### ノートブックでライブ W&B インターフェイスを直接レンダリング

既存のダッシュボード、スイープ、またはレポートをノートブック内で直接表示することも可能です。`%wandb` マジックを使います：

```notebook
# プロジェクトワークスペースを表示
%wandb USERNAME/PROJECT
# 単一の run を表示
%wandb USERNAME/PROJECT/runs/RUN_ID
# スイープを表示
%wandb USERNAME/PROJECT/sweeps/SWEEP_ID
# レポートを表示
%wandb USERNAME/PROJECT/reports/REPORT_ID
# 埋め込まれた iframe の高さを指定
%wandb USERNAME/PROJECT -h 2048
```

`%%wandb` または `%wandb` マジックの代替として、`wandb.init()` を実行した後、任意のセルを `wandb.run` で終わらせてインライングラフを表示するか、私たちの API から返された任意のレポート、スイープ、または run オブジェクトに `ipython.display(...)` を呼び出すこともできます。

```python
# まず wandb.run を初期化
wandb.init()

# セルが wandb.run を出力すれば、ライブグラフが見られます
wandb.run
```

{{% alert %}}
W&B でできることについてもっと知りたいですか？[データとメディアのログガイド]({{< relref path="/guides/models/track/log/" lang="ja" >}})をチェックし、[お気に入りの ML ツールキットとのインテグレーション方法]({{< relref path="/guides/integrations/" lang="ja" >}})を学ぶか、[リファレンスドキュメント]({{< relref path="/ref/python/" lang="ja" >}})または私たちの[例のレポジトリ](https://github.com/wandb/examples)に直接飛び込んでください。
{{% /alert %}}

## W&B における追加の Jupyter 機能

1. **Colab における簡単な認証**: Colab で最初に `wandb.init` を呼び出すと、ブラウザーで W&B にログインしている場合、ランタイムを自動的に認証します。run ページの Overviewタブに Colab へのリンクが表示されます。
2. **Jupyter マジック**: ダッシュボード、スイープ、レポートをノートブック内で直接表示する機能です。`%wandb` マジックはプロジェクト、スイープ、またはレポートへのパスを受け取り、W&B インターフェイスをノートブック内に直接レンダリングします。
3. **Docker化された Jupyter のローンチ**: `wandb docker --jupyter` を呼び出して、dockerコンテナを起動し、コードをマウントし、Jupyter がインストールされていることを確認し、ポート 8888 で起動します。
4. **順序を気にせずにセルを実行**: デフォルトでは、次に `wandb.init` が呼び出されるまで run を `finished` としてマークしません。それにより、複数のセル（例: データを設定するセル、トレーニングするセル、テストするセル）を任意の順序で実行し、すべて同じ run にログできます。[設定](https://app.wandb.ai/settings)でコード保存をオンにすると、実行順序と状態で実行されたセルもログされ、最も非線形なパイプラインでさえ再現できます。Jupyter ノートブックで run を手動で完了としてマークするには、`run.finish` を呼び出してください。

```python
import wandb

run = wandb.init()

# トレーニングスクリプトとログはここに

run.finish()
```