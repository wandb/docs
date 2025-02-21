---
title: Sweeps
description: W&B Sweeps を使用したハイパーパラメータ ーサーチとモデルの最適化
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

W&B Sweeps を使用して、ハイパーパラメータの検索を自動化し、豊富なインタラクティブな 実験管理 を視覚化します。ベイズ、 グリッド検索 、ランダムなどの一般的な検索 method から選択して、ハイパーパラメータ空間を検索します。1つまたは複数のマシンにまたがって sweep をスケールおよび並列化します。

{{< img src="/images/sweeps/intro_what_it_is.png" alt="インタラクティブなダッシュボードで、大規模なハイパーパラメータ チューニング 実験から洞察を引き出します。" >}}

### 仕組み
2つの [W&B CLI]({{< relref path="/ref/cli/" lang="ja" >}}) コマンドで sweep を作成します。

1. sweep を初期化します。

```bash
wandb sweep --project <propject-name> <path-to-config file>
```

2. sweep agent を起動します。

```bash
wandb agent <sweep-ID>
```

{{% alert %}}
上記のコードスニペット、およびこのページにリンクされている colab は、W&B CLI を使用して sweep を初期化および作成する方法を示しています。sweep configuration を定義し、sweep を初期化し、sweep を開始するために使用する W&B Python SDK コマンドのステップごとの概要については、Sweeps の [Walkthrough]({{< relref path="./walkthrough.md" lang="ja" >}}) を参照してください。
{{% /alert %}}

### 開始方法

ユースケースに応じて、次のリソースを調べて W&B Sweeps を開始してください。

* W&B Python SDK コマンドを使用して、sweep configuration を定義し、sweep を初期化し、sweep を開始する方法のステップごとの概要については、[sweeps walkthrough]({{< relref path="./walkthrough.md" lang="ja" >}}) をお読みください。
* この チャプター では、次の方法について説明します。
  * [W&B を コード に追加する]({{< relref path="./add-w-and-b-to-your-code.md" lang="ja" >}})
  * [sweep configuration を定義する]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ja" >}})
  * [sweeps を初期化する]({{< relref path="./initialize-sweeps.md" lang="ja" >}})
  * [sweep agents を起動する]({{< relref path="./start-sweep-agents.md" lang="ja" >}})
  * [sweep results を視覚化する]({{< relref path="./visualize-sweep-results.md" lang="ja" >}})
* W&B Sweeps を使用したハイパーパラメータ最適化を調査する [厳選された Sweep experiments のリスト]({{< relref path="./useful-resources.md" lang="ja" >}}) を調べてください。results は W&B Reports に保存されます。

ステップごとのビデオについては、[Tune Hyperparameters Easily with W&B Sweeps](https://www.youtube.com/watch?v=9zrmUIlScdY\&ab_channel=Weights%26Biases) をご覧ください。
