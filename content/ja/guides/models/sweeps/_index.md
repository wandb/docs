---
title: スイープ
description: W&B スイープによるハイパーパラメーター探索とモデル最適化
cascade:
- url: /ja/guides/sweeps/:filename
menu:
  default:
    identifier: ja-guides-models-sweeps-_index
    parent: w-b-models
url: /ja/guides/sweeps
weight: 2
---

{{< cta-button productLink="https://wandb.ai/stacey/deep-drive/workspace?workspace=user-lavanyashukla" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb" >}}

W&B Sweeps を使用してハイパーパラメータ検索を自動化し、豊富でインタラクティブな実験管理を視覚化します。ベイズ、グリッド検索、ランダムなどの一般的な検索メソッドから選択して、ハイパーパラメータ空間を探索できます。スイープを 1 台以上のマシンにわたってスケールし、並列化します。

{{< img src="/images/sweeps/intro_what_it_is.png" alt="インタラクティブなダッシュボードで大規模なハイパーパラメータチューニング実験からインサイトを引き出します。" >}}

### 仕組み
2 つの [W&B CLI]({{< relref path="/ref/cli/" lang="ja" >}}) コマンドで sweep を作成します:

1. スイープを初期化する

```bash
wandb sweep --project <propject-name> <path-to-config file>
```

2. スイープエージェントを開始する

```bash
wandb agent <sweep-ID>
```

{{% alert %}}
上記のコードスニペットとこのページにリンクされている colab は、W&B CLI でスイープを初期化し作成する方法を示しています。W&B Python SDK コマンドを使用してスイープ設定を定義し、スイープを初期化・開始する手順については Sweeps の[ウォークスルー]({{< relref path="./walkthrough.md" lang="ja" >}})をご覧ください。
{{% /alert %}}

### 開始方法

ユースケースに応じて、W&B Sweeps の開始に役立つ次のリソースを探索してください:

* スイープ設定を定義し、スイープを初期化・開始するための W&B Python SDK コマンドの手順については、[スイープのウォークスルー]({{< relref path="./walkthrough.md" lang="ja" >}})をお読みください。
* 次のチャプターを探索して以下を学びます:
  * [W&B をコードに追加する]({{< relref path="./add-w-and-b-to-your-code.md" lang="ja" >}})
  * [スイープ設定を定義する]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ja" >}})
  * [スイープを初期化する]({{< relref path="./initialize-sweeps.md" lang="ja" >}})
  * [スイープエージェントを開始する]({{< relref path="./start-sweep-agents.md" lang="ja" >}})
  * [スイープ結果を視覚化する]({{< relref path="./visualize-sweep-results.md" lang="ja" >}})
* W&B Sweeps を使用したハイパーパラメータ最適化を探索する [スイープ実験の厳選リスト]({{< relref path="./useful-resources.md" lang="ja" >}})を探索します。結果は W&B Reports に保存されます。

ステップバイステップのビデオについては、こちらをご覧ください: [Tune Hyperparameters Easily with W&B Sweeps](https://www.youtube.com/watch?v=9zrmUIlScdY\&ab_channel=Weights%26Biases).