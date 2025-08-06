---
title: Jupyter ノートブックをトラッキングする
description: Jupyter と W&B を使うと、ノートブックから離れることなくインタラクティブな可視化が利用できます。
menu:
  default:
    identifier: jupyter
    parent: experiments
weight: 6
---

Jupyter と W&B を組み合わせて、ノートブックから離れることなくインタラクティブな可視化を実現しましょう。カスタム分析、実験、プロトタイプなどを自由に組み合わせ、それらをすべて完全にログできます。

## Jupyter ノートブックでの W&B のユースケース

1. **反復的な実験**: 実験を何度も実行し、パラメータを調整しても、その都度手動で記録を取ることなく、すべての run が自動的に W&B に保存されます。
2. **コードの保存**: モデルを再現しようとする時、ノートブック内でどのセルがどの順番で実行されたかを把握するのは大変です。[設定ページ]({{< relref "/guides/models/app/settings-page/" >}}) でコード保存を有効にすると、各実験におけるセル実行の記録を残せます。
3. **カスタム分析**: run が W&B に記録されたら、API から簡単に dataframe を取得してカスタム分析ができます。その分析結果も W&B にログすれば、データや可視化を Reports に保存・共有できます。

## ノートブックでのはじめ方

まず、以下のコードで W&B をインストールし、あなたのアカウントとリンクします。

```notebook
!pip install wandb -qqq
import wandb
wandb.login()
```

次に、実験のセットアップとハイパーパラメーターの保存を行いましょう。

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

`wandb.init()` を実行したあとは、新しいセルで `%%wandb` を使うとノートブック内でライブグラフを確認できます。このセルを複数回実行すると、その都度データが run に追加されます。

```notebook
%%wandb

# ここにトレーニングループを記述
```

実際にこちらの [サンプルノートブック](https://wandb.me/jupyter-interact-colab) でお試しください。

{{< img src="/images/track/jupyter_widget.png" alt="Jupyter W&B ウィジェット" >}}

### ノートブック内で W&B のライブインターフェースを表示する

ノートブックで `%wandb` マジックを使うことで、既存のダッシュボード、スイープ、Reports などを簡単に表示できます。

```notebook
# Project の workspace を表示
%wandb USERNAME/PROJECT
# 単一の run を表示
%wandb USERNAME/PROJECT/runs/RUN_ID
# スイープを表示
%wandb USERNAME/PROJECT/sweeps/SWEEP_ID
# Report を表示
%wandb USERNAME/PROJECT/reports/REPORT_ID
# 埋め込み iframe の高さを指定
%wandb USERNAME/PROJECT -h 2048
```

`%%wandb` や `%wandb` マジックの代わりに、 `wandb.init()` の後、任意のセルの最後で `wandb.Run.finish()` を実行すればインラインでグラフが表示されます。また、API から取得した report, sweep, run オブジェクトを `ipython.display(...)` で表示することも可能です。

```python
import wandb
from IPython.display import display
# run を初期化
run = wandb.init()

# セルの出力として run.finish() を記述するとライブグラフが表示されます
run.finish()
```

{{% alert %}}
W&B でできることをもっと知りたい方は、[データやメディアのログ方法ガイド]({{< relref "/guides/models/track/log/" >}})、[お気に入りの ML ツールキットとの連携方法]({{< relref "/guides/integrations/" >}})、[リファレンスドキュメント]({{< relref "/ref/python/" >}})、あるいは [サンプルリポジトリ](https://github.com/wandb/examples) もぜひご覧ください。
{{% /alert %}}

## W&B の Jupyter 向け追加機能

1. **Colab での簡単認証**: Colab で初めて `wandb.init` を実行すると、ブラウザで W&B にログイン済みならランタイムが自動で認証されます。run ページの Overviewタブ には Colab へのリンクも表示されます。
2. **Jupyter マジック**: ダッシュボード、スイープ、Reports をノートブック内で直接表示できます。 `%wandb` マジックで project, sweeps, reports のパスを指定するだけで、W&B インターフェースがノートブック内にそのままレンダリングされます。
3. **Jupyter を docker で起動**: `wandb docker --jupyter` で dockerコンテナ を起動し、あなたのコードをマウント、Jupyter をインストール、さらにポート 8888 で起動可能です。
4. **セルを任意の順序で安心して実行**: デフォルトでは `wandb.init` が次に呼ばれるまで run を `finished` としません。つまり、データセット準備、トレーニング、テストといった複数のセルを好きな順に実行しても、すべて同じ run にログできます。[設定](https://app.wandb.ai/settings) でコード保存を有効にすれば、実行したセルとその時の状態も記録でき、複雑なパイプラインでも再現が容易です。Jupyter ノートブックで run を手動で完了するには、`run.finish` を呼び出してください。

```python
import wandb

run = wandb.init()

# トレーニングスクリプトやログ処理をここに記述

run.finish()
```