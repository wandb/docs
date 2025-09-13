---
title: Jupyter ノートブックを追跡する
description: Jupyter と W&B を使えば、ノートブックを離れることなくインタラクティブな可視化を利用できます。
menu:
  default:
    identifier: ja-guides-models-track-jupyter
    parent: experiments
weight: 6
---

ノートブックから離れずに、W&B を Jupyter と組み合わせてインタラクティブな可視化を行いましょう。カスタム分析、実験、プロトタイプを組み合わせ、すべてを完全にログできます。

## Jupyter ノートブックでの W&B のユースケース

1. Iterative experimentation: 実験を何度も実行してパラメータを調整し、そのたびに手動でメモを取らなくても、実行した run が自動的に W&B に保存されます。
2. Code saving: ノートブックでモデルを再現する際、どのセルがどの順番で実行されたかを把握するのは大変です。[設定ページ]({{< relref path="/guides/models/app/settings-page/" lang="ja" >}}) でコード保存を有効にすると、各実験におけるセル実行の記録を保存できます。
3. Custom analysis: run を W&B にログしたら、API からデータフレームを取得してカスタム分析を行い、その結果を W&B にログして保存し、Reports で共有するのも簡単です。

## ノートブックでのはじめ方

以下のコードで W&B をインストールし、アカウントにリンクしてノートブックを開始します:

```notebook
!pip install wandb -qqq
import wandb
wandb.login()
```

次に、実験をセットアップしてハイパーパラメーターを保存します:

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

`wandb.init()` を実行したら、ノートブックで `%%wandb` から始まる新しいセルを作成すると、ライブグラフが表示されます。このセルを複数回実行すると、データは同じ run に追記されます。

```notebook
%%wandb

# ここにトレーニングループを記述
```

この [サンプルノートブック](https://wandb.me/jupyter-interact-colab) で試してみてください。

{{< img src="/images/track/jupyter_widget.png" alt="Jupyter の W&B ウィジェット" >}}

### ノートブック内で W&B のライブインターフェースを直接レンダリングする

`%wandb` マジックを使うと、既存のダッシュボード、sweeps、reports をノートブック内に直接表示できます:

```notebook
# Project の Workspace を表示
%wandb USERNAME/PROJECT
# 単一の run を表示
%wandb USERNAME/PROJECT/runs/RUN_ID
# sweep を表示
%wandb USERNAME/PROJECT/sweeps/SWEEP_ID
# report を表示
%wandb USERNAME/PROJECT/reports/REPORT_ID
# 埋め込み iframe の高さを指定
%wandb USERNAME/PROJECT -h 2048
```

`%%wandb` や `%wandb` マジックの代わりに、`wandb.init()` 実行後であれば、任意のセルの末尾を `wandb.Run.finish()` にしてインラインのグラフを表示したり、API から返される任意の report、sweep、run オブジェクトに対して `ipython.display(...)` を呼び出すこともできます。

```python
import wandb
from IPython.display import display
# run を初期化
run = wandb.init()

# このセルが run.finish() を出力すると、ライブグラフが表示されます
run.finish()
```

{{% alert %}}
W&B でできることをもっと知りたいですか？[データとメディアのログ方法ガイド]({{< relref path="/guides/models/track/log/" lang="ja" >}})、[お好みの ML ツールキットとの連携方法]({{< relref path="/guides/integrations/" lang="ja" >}}) をチェックするか、[リファレンスドキュメント]({{< relref path="/ref/python/" lang="ja" >}}) や [サンプル集のリポジトリ](https://github.com/wandb/examples) を直接のぞいてみてください。
{{% /alert %}}

## W&B における Jupyter 向けの追加機能

1. Easy authentication in Colab: Colab で初めて `wandb.init` を呼び出すと、ブラウザで W&B にログイン済みであれば、ランタイムを自動的に認証します。run ページの Overview タブに、Colab へのリンクが表示されます。
2. Jupyter Magic: ダッシュボード、sweeps、reports をノートブック内に直接表示します。`%wandb` マジックは、project、sweeps、reports へのパスを受け取り、W&B のインターフェースをノートブック内に直接レンダリングします。
3. Launch dockerized Jupyter: `wandb docker --jupyter` を呼び出すと、docker コンテナを起動してそこにコードをマウントし、Jupyter がインストールされていることを確実にした上で、ポート 8888 で起動します。
4. Run cells in arbitrary order without fear: 既定では、次に `wandb.init` が呼び出されるまで run を `finished` としてマークしません。これにより、複数のセル（たとえばデータのセットアップ、学習、テスト）を好きな順序で実行しても、同じ run にログできます。さらに [settings](https://app.wandb.ai/settings) でコード保存を有効にすると、実行されたセルを、その実行順序と実行時の状態とともにログするため、非常に非線形なパイプラインでも再現可能になります。Jupyter ノートブックで手動で run を完了にしたい場合は、`run.finish` を呼び出してください。

```python
import wandb

run = wandb.init()

# ここにトレーニングスクリプトと ログ を記述

run.finish()
```