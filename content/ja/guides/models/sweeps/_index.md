---
title: Sweeps
description: W&B Sweeps でハイパーパラメーター探索とモデル最適化
menu:
  default:
    identifier: sweeps
    parent: w-b-models
url: guides/sweeps
weight: 2
cascade:
- url: guides/sweeps/:filename
---

{{< cta-button productLink="https://wandb.ai/stacey/deep-drive/workspace?workspace=user-lavanyashukla" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb" >}}

W&B Sweeps を使えば、ハイパーパラメータ探索を自動化し、豊富でインタラクティブな実験管理を可視化できます。Bayesian、グリッド検索、ランダムなどの一般的な探索手法を選び、ハイパーパラメータ空間を効率的に探索しましょう。Sweep を一台または複数マシンでスケール・並列実行できます。

{{< img src="/images/sweeps/intro_what_it_is.png" alt="ハイパーパラメータチューニングのインサイト" >}}

### 仕組み
2つの [W&B CLI]({{< relref "/ref/cli/" >}}) コマンドで Sweep を作成します。

1. Sweep の初期化

```bash
wandb sweep --project <project-name> <path-to-config file>
```

2. Sweep agent の起動

```bash
wandb agent <sweep-ID>
```

{{% alert %}}
上記のコードスニペットと、このページで紹介している colab では、W&B CLI を使った Sweep の初期化・実行方法を紹介しています。Sweep 設定の定義から初期化、起動までを段階的に説明している [Sweeps walkthrough]({{< relref "./walkthrough.md" >}}) もご参照ください。
{{% /alert %}}


### 開始方法

ユースケースに応じて、以下のリソースを活用しながら W&B Sweeps を始められます。

* Sweep 設定の定義から初期化、開始までを段階的に解説した [sweeps walkthrough]({{< relref "./walkthrough.md" >}}) をご覧ください。
* このチャプターでは次の内容が学べます：
  * [W&B をコードに追加する]({{< relref "./add-w-and-b-to-your-code.md" >}})
  * [Sweep 設定を定義する]({{< relref "/guides/models/sweeps/define-sweep-configuration/" >}})
  * [Sweeps を初期化する]({{< relref "./initialize-sweeps.md" >}})
  * [Sweep agent を開始する]({{< relref "./start-sweep-agents.md" >}})
  * [Sweep の結果を可視化する]({{< relref "./visualize-sweep-results.md" >}})
* ハイパーパラメータ最適化に W&B Sweeps を活用した [Sweep Experiments の厳選リスト]({{< relref "./useful-resources.md" >}}) もご参照ください。結果は W&B Reports に保存されます。

操作手順を動画で見たい方は、こちらをご覧ください: [Tune Hyperparameters Easily with W&B Sweeps](https://www.youtube.com/watch?v=9zrmUIlScdY\&ab_channel=Weights%26Biases)。