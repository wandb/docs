---
title: Sweeps
description: W&B Sweeps によるハイパーパラメーター探索とモデル最適化
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


W&B Sweeps を使ってハイパーパラメーター探索を自動化し、リッチでインタラクティブな 実験管理 を可視化しましょう。ベイズ、グリッド検索、ランダムなどの一般的な探索手法から選んでハイパーパラメーター空間を探索できます。1 台以上のマシンにまたがって sweep をスケールさせ、並列化できます。

{{< img src="/images/sweeps/intro_what_it_is.png" alt="ハイパーパラメータチューニングのインサイト" >}}

### 仕組み
2 つの [W&B CLI]({{< relref path="/ref/cli/" lang="ja" >}}) コマンドで sweep を作成します:

1. sweep を初期化する

```bash
wandb sweep --project <project-name> <path-to-config file>
```

2. sweep agent を起動する

```bash
wandb agent <sweep-ID>
```

{{% alert %}}
上のコードスニペットと、このページからリンクされている Colab は、W&B CLI を使って sweep を初期化・作成する方法を示しています。段階的な手順については、[Sweeps ウォークスルー]({{< relref path="./walkthrough.md" lang="ja" >}}) を参照してください。ここでは W&B Python SDK のコマンドを使って、sweep configuration を定義し、sweep を初期化し、開始する方法を説明しています。
{{% /alert %}}



### 開始方法

ユースケースに応じて、以下のリソースを参照して W&B Sweeps を使い始めましょう:

* [Sweeps ウォークスルー]({{< relref path="./walkthrough.md" lang="ja" >}}) を読んで、W&B Python SDK のコマンドを使って sweep configuration を定義し、sweep を初期化して開始するまでの手順を確認してください。
* このチャプターでは次のことを学べます:
  * [コードに W&B を追加する]({{< relref path="./add-w-and-b-to-your-code.md" lang="ja" >}})
  * [sweep configuration を定義する]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ja" >}})
  * [sweep を初期化する]({{< relref path="./initialize-sweeps.md" lang="ja" >}})
  * [sweep agent を開始する]({{< relref path="./start-sweep-agents.md" lang="ja" >}})
  * [sweep の結果を可視化する]({{< relref path="./visualize-sweep-results.md" lang="ja" >}})
* W&B Sweeps でハイパーパラメータチューニングに取り組んだ、[厳選された Sweep experiments の一覧]({{< relref path="./useful-resources.md" lang="ja" >}}) をチェックしてください。結果は W&B Reports に保存されています。

ステップバイステップの動画は次をご覧ください: [W&B Sweeps でハイパーパラメーターを簡単にチューニングする](https://www.youtube.com/watch?v=9zrmUIlScdY\&ab_channel=Weights%26Biases)