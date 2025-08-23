---
title: Sweeps
description: W&B Sweeps でのハイパーパラメーター探索とモデル最適化
cascade:
- url: guides/sweeps/:filename
menu:
  default:
    identifier: ja-guides-models-sweeps-_index
    parent: w-b-models
url: guides/sweeps
weight: 2
---

{{< cta-button productLink="https://wandb.ai/stacey/deep-drive/workspace?workspace=user-lavanyashukla" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb" >}}

W&B Sweeps を使うことで、ハイパーパラメータの探索を自動化し、リッチでインタラクティブな実験管理を可視化できます。Bayesian、グリッド検索、ランダムなどの代表的な探索手法から選択して、ハイパーパラメータ空間を探索できます。sweep を 1 台または複数台のマシンにスケール・並列化することも可能です。

{{< img src="/images/sweeps/intro_what_it_is.png" alt="ハイパーパラメータチューニングのインサイト" >}}

### 仕組み
2つの [W&B CLI]({{< relref path="/ref/cli/" lang="ja" >}}) コマンドで sweep を作成します。

1. sweep を初期化

```bash
wandb sweep --project <project-name> <path-to-config file>
```

2. sweep agent を起動

```bash
wandb agent <sweep-ID>
```

{{% alert %}}
上記のコードスニペットおよび本ページで紹介されている Colab では、W&B CLI を使った sweep の初期化・作成方法を示しています。sweep 設定の定義、初期化、開始を段階的に解説した [Sweeps walkthrough]({{< relref path="./walkthrough.md" lang="ja" >}}) も参考にしてください。
{{% /alert %}}

### 開始方法

ユースケースに応じて、W&B Sweeps の活用を始めるために以下のリソースを参照してください。

* sweep 設定の定義・初期化・開始までを細かく解説した [sweeps walkthrough]({{< relref path="./walkthrough.md" lang="ja" >}}) を読む
* このチャプターで次のことを学ぶ:
  * [コードに W&B を追加]({{< relref path="./add-w-and-b-to-your-code.md" lang="ja" >}})
  * [sweep 設定の定義]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ja" >}})
  * [Sweep の初期化]({{< relref path="./initialize-sweeps.md" lang="ja" >}})
  * [sweep agent の開始]({{< relref path="./start-sweep-agents.md" lang="ja" >}})
  * [sweep 結果の可視化]({{< relref path="./visualize-sweep-results.md" lang="ja" >}})
* W&B Sweeps でハイパーパラメータ最適化を行った [厳選された Sweep Experiments のリスト]({{< relref path="./useful-resources.md" lang="ja" >}}) を参照することもできます。結果は W&B Reports に保存されます。

ステップバイステップの動画もご覧ください：[Tune Hyperparameters Easily with W&B Sweeps](https://www.youtube.com/watch?v=9zrmUIlScdY\&ab_channel=Weights%26Biases)。