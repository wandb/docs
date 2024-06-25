---
description: W&Bで機械学習実験をトラックする。
slug: /guides/track
displayed_sidebar: default
---

import Translate, {translate} from '@docusaurus/Translate';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';


# 実験の管理

<CTAButtons productLink="https://wandb.ai/stacey/deep-drive/workspace?workspace=user-lavanyashukla" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases.ipynb"/>

<head>
  <title>機械学習とディープラーニングの実験を追跡する</title>
</head>

数行のコードで機械学習実験を追跡します。結果は [インタラクティブなダッシュボード](app.md) で確認することも、Pythonにデータをエクスポートしてプログラムによるアクセスを行うこともできます（[Public API](../../ref/python/public-api/README.md) を使用）。

人気のあるフレームワークを使用している場合は、W&B Integrations を活用してください。[PyTorch](../integrations/pytorch.md)、[Keras](../integrations/keras.md)、または [Scikit](../integrations/scikit.md)などです。[Integration guides](../integrations/intro.md) で、インテグレーションの完全なリストとコードにW&Bを追加する方法に関する情報を確認できます。

![](/images/experiments/experiments_landing_page.png)

上の画像は、複数の[ラン](../runs/intro.md)にわたってメトリクスを表示および比較できるダッシュボードの例を示しています。

## 仕組み

数行のコードを使って機械学習実験を追跡します:
1. [W&B run](../runs/intro.md) を作成。
2. 学習率やモデルタイプなどのハイパーパラメータを辞書形式で設定 ([`wandb.config`](./config.md))。
3. トレーニングループの中で精度や損失などのメトリクスをログ ([`wandb.log()`](./log/intro.md))。
4. モデルの重みや予測のテーブルなど、runの結果を保存。

以下の疑似コードでは、一般的なW&B実験管理ワークフローを示しています：

```python showLineNumbers
# 1. W&B Runを開始
wandb.init(entity="", project="my-project-name")

# 2. モデルの入力とハイパーパラメータを保存
wandb.config.learning_rate = 0.01

# モデルとデータをインポート
model, dataloader = get_model(), get_data()

# モデルトレーニングのコードがここに入ります

# 3. メトリクスをログしてパフォーマンスを可視化
wandb.log({"loss": loss})

# 4. モデルをW&Bにアーティファクトとしてログ
wandb.log_artifact(model)
```

## 開始方法

ユースケースに応じて、W&B Experimentsを始めるための次のリソースを探索してください：

* 初めてW&B Artifactsを使用する場合は、[Experiments Colabノートブック](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases.ipynb) をご覧ください。
* [W&B クイックスタート](../../quickstart.md) を読んで、W&B Python SDKコマンドを使用してデータセットアーティファクトを作成、追跡、使用するためのステップバイステップガイドを参照してください。
* このチャプターを探索して、以下を学びましょう：
  * 実験の作成方法
  * 実験の設定方法
  * 実験からデータをログする方法
  * 実験結果を表示する方法
* [W&B Pythonライブラリ](../../ref/python/README.md) を [W&B APIリファレンスガイド](../../ref/README.md) 内で探索。