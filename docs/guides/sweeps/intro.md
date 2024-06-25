---
description: W&B Sweeps を使用したハイパーパラメータ検索とモデル最適化
slug: /guides/sweeps
displayed_sidebar: default
---

import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';


# ハイパーパラメータのチューニング

<CTAButtons productLink="https://wandb.ai/stacey/deep-drive/workspace?workspace=user-lavanyashukla" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb"/>

<head>
  <title>Sweepsでハイパーパラメータをチューニングする</title>
</head>

W&B Sweepsを使用してハイパーパラメータの検索を自動化し、リッチでインタラクティブな実験管理を視覚化します。ベイズ法、グリッド検索、ランダム検索などの人気のある検索方法から選び、ハイパーパラメータ空間を検索します。複数のマシンに渡ってスケールし、並列化することができます。

![インタラクティブなダッシュボードを使用して、大規模なハイパーパラメータチューニング実験から洞察を得る。](/images/sweeps/intro_what_it_is.png)

### 仕組み
2つの [W&B CLI](../../ref/cli/README.md) コマンドでsweepを作成します：

1. sweepを初期化する

```bash
wandb sweep --project <propject-name> <path-to-config file>
```

2. sweep agentを開始する

```bash
wandb agent <sweep-ID>
```

:::tip
上記のコードスニペットとこのページにリンクされているColabでは、W&B CLIを使用してsweepを初期化し作成する方法を示しています。詳細な手順については、Sweepsの [ウォークスルー](./walkthrough.md) を参照してください。W&B Python SDKコマンドを使用してsweep設定を定義、初期化、開始する方法がステップバイステップで説明されています。
:::

### 開始方法

ユースケースに応じて、以下のリソースを参照してW&B Sweepsを始めてください：

* 初めてW&B Sweepsを使用する場合は、[Sweeps Colab notebook](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb) を参照することをお勧めします。
* W&B Python SDKコマンドを使用してsweep設定を定義、初期化、開始する方法の詳細については、[sweeps walkthrough](./walkthrough.md) を読んでください。
* このチャプターでは以下の方法を学びます：
  * [コードにW&Bを追加する](./add-w-and-b-to-your-code.md)
  * [sweep設定を定義する](./define-sweep-configuration.md)
  * [sweepsを初期化する](./initialize-sweeps.md)
  * [sweep agentsを開始する](./start-sweep-agents.md)
  * [sweep結果を視覚化する](./visualize-sweep-results.md)
* W&B Sweepsを使用してハイパーパラメータ最適化を探求する [キュレーションされたSweep実験のリスト](./useful-resources.md) を参照してください。結果はW&B Reportsに保存されます。

ステップバイステップのビデオは以下を参照してください：[Tune Hyperparameters Easily with W&B Sweeps](https://www.youtube.com/watch?v=9zrmUIlScdY\&ab\_channel=Weights%26Biases).