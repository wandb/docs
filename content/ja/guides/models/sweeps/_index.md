---
title: Sweeps
description: W&B Sweeps を使用したハイパーパラメータ探索と モデル の最適化
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

W&B Sweeps を使用して、ハイパーパラメータの検索を自動化し、豊富なインタラクティブな 実験管理 を視覚化します。ベイズ、グリッド検索、ランダムなどの一般的な検索メソッドから選択して、ハイパーパラメータ空間を検索します。1つまたは複数のマシンに スイープ をスケールおよび並列化します。

{{< img src="/images/sweeps/intro_what_it_is.png" alt="インタラクティブなダッシュボードで、大規模なハイパーパラメータチューニング実験から洞察を引き出します。" >}}

### 仕組み
2つの [W&B CLI]({{< relref path="/ref/cli/" lang="ja" >}}) コマンドで スイープ を作成します。

1. スイープ を初期化する

```bash
wandb sweep --project <propject-name> <path-to-config file>
```

2. sweep agent を起動する

```bash
wandb agent <sweep-ID>
```

{{% alert %}}
上記の コードスニペット と、このページにリンクされている Colab は、W&B CLI で スイープ を初期化および作成する方法を示しています。 スイープ設定を定義し、 スイープ を初期化し、 スイープ を開始するために使用する W&B Python SDK コマンドのステップごとの概要については、Sweeps の [Walkthrough]({{< relref path="./walkthrough.md" lang="ja" >}}) を参照してください。
{{% /alert %}}

### 開始方法

ユースケース に応じて、次のリソースを参照して W&B Sweeps を開始してください。

* スイープ設定を定義し、 スイープ を初期化し、 スイープ を開始するために使用する W&B Python SDK コマンドのステップごとの概要については、[sweeps walkthrough]({{< relref path="./walkthrough.md" lang="ja" >}}) を参照してください。
* この chapter では、次の方法について説明します。
  * [W&B を code に追加]({{< relref path="./add-w-and-b-to-your-code.md" lang="ja" >}})
  * [スイープ 設定を定義]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ja" >}})
  * [スイープ を初期化]({{< relref path="./initialize-sweeps.md" lang="ja" >}})
  * [sweep agent を起動]({{< relref path="./start-sweep-agents.md" lang="ja" >}})
  * [スイープ の result を視覚化]({{< relref path="./visualize-sweep-results.md" lang="ja" >}})
* W&B Sweeps でハイパーパラメータ最適化を調査する [厳選された Sweep experiments のリスト]({{< relref path="./useful-resources.md" lang="ja" >}}) を見てみましょう。 result は W&B Reports に保存されます。

ステップごとのビデオについては、[Tune Hyperparameters Easily with W&B Sweeps](https://www.youtube.com/watch?v=9zrmUIlScdY\&ab_channel=Weights%26Biases) を参照してください。
