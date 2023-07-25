---
description: >-
  An overview of what is Weights & Biases along with links on how to get started
  if you are a first time user.
slug: /guides
displayed_sidebar: ja
---

# Weights & Biases とは？

Weights & Biasesは、開発者がより優れたモデルを迅速に構築できる機械学習プラットフォームです。W&Bの軽量で相互運用可能なツールを使用して、すばやく実験をトラッキングし、データセットのバージョン管理と反復処理を行い、モデルのパフォーマンスを評価し、モデルを再現し、結果を視覚化して品質の低下を検出し、同僚との発見を共有します。W&Bを5分で設定し、データセットとモデルが信頼性のある記録システムでトラッキングされ、バージョン管理されていることを確信して、機械学習パイプラインをすばやく反復処理します。

<!-- ![](@site/static/images/general/diagram_2021.png) -->

## W&Bの初めてのユーザーですか？

もし、初めてW&Bを使う場合は、以下を試してみてください。

1. Weights & Biases を使ってみる: [run an example introduction project with Google Colab](http://wandb.me/intro).
1. [Quickstart](../quickstart.md) を読んで、W&Bをコードに追加する方法と場所の概要を把握してください。
1. [Weights & Biasesはどのように動作しますか？](#how-does-weights--biases-work) このセクションでは、W&Bの構成要素について概説しています。
1. [Integrations guide](./integrations/intro.md) や [W&B Easy Integration YouTube](https://www.youtube.com/playlist?list=PLD80i8An1OEGDADxOBaH71ZwieZ9nmPGC) を参照して、お好みの機械学習フレームワークとW&Bを統合する方法について情報を得ることができます。
1. [API Reference guide](../ref/README.md) には、W&B Pythonライブラリ、CLI、Weave操作に関する技術仕様が記載されています。

## Weights & Biasesはどのように動作しますか？

W&Bの初めてのユーザーであれば、以下のセクションをこの順番に読むことをお勧めします。

1. W&Bの基本的な計算単位である [Runs](./runs/intro.md) について学びます。
2. [Experiments](./track/intro.md) を使って機械学習の実験を作成し、トラッキングします。
3. データセットとモデルのバージョン管理の柔軟で軽量なビルディングブロック、[Artifacts](./artifacts/intro.md) について学びます。
4. [Sweeps](./sweeps/intro.md) を使ってハイパーパラメーターの探索を自動化し、可能性のあるモデルの空間を探索します。
5. [Model Management](./models/intro.md) を使って、モデルのライフサイクルをトレーニングからプロダクションまで管理します。
6. [Data Visualization](./tables/intro.md) ガイドで、モデルバージョン間の予測を視覚化します。
7. [Reports](./reports/intro.md) を使って、W&BのRunを整理し、視覚化を埋め込み、自動化し、発見を説明し、共同作業者とアップデートを共有します。