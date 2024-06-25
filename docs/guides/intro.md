---
description: W&B の概要、および初めてのユーザー向けの開始方法に関するリンク。
slug: /guides
displayed_sidebar: default
---


# W&Bとは?

Weights & Biases (W&B) はAI開発者向けプラットフォームで、モデルのトレーニング、ファインチューニング、そして基盤モデルの活用のためのツールを提供します。

W&Bを5分でセットアップし、実験パイプラインに素早く繰り返し取り組むことで、モデルとデータが信頼できるシステムで追跡・バージョン管理されている安心感を得られます。

![](@site/static/images/general/architecture.png)

この図はW&B製品間の関係を示しています。

**[W&B Models](/guides/models.md)** は、機械学習エンジニアがモデルをトレーニング・ファインチューニングするための軽量で相互運用可能なツールセットです。
- [Experiments](/guides/track/intro.md): 機械学習実験の追跡
- [Model Registry](/guides/model_registry/intro.md): プロダクションモデルを中央管理
- [Launch](/guides/launch/intro.md): ワークロードのスケールと自動化
- [Sweeps](/guides/sweeps/intro.md): ハイパーパラメータチューニングとモデル最適化

**[W&B Weave](https://wandb.github.io/weave/)** はLLMアプリケーションの追跡と評価のための軽量ツールキットです。

**[W&B Core](/guides/platform.md)** はデータとモデルの追跡と可視化、そして結果の共有のための強力な基盤要素です。
- [Artifacts](/guides/artifacts/intro.md): アセットのバージョン管理とリネージの追跡
- [Tables](/guides/tables/intro.md): 表形式データの可視化とクエリ
- [Reports](/guides/reports/intro.md): 発見を文書化し、共同作業

## 初めてW&Bを使うユーザーですか？

<iframe width="100%" height="330" src="https://www.youtube.com/embed/tHAFujRhZLA" title="Weights &amp; Biases End-to-End Demo" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

これらのリソースを使ってW&Bを始めてみましょう:

1. [Intro Notebook](http://wandb.me/intro): 短いサンプルコードを実行して5分で実験を追跡
2. [Quickstart](../quickstart.md): W&Bをコードに追加する方法と場所の概要
3. [インテグレーションガイド](./integrations/intro.md)と[W&B Easy Integration YouTube](https://www.youtube.com/playlist?list=PLD80i8An1OEGDADxOBaH71ZwieZ9nmPGC) プレイリストを探索し、お好みの機械学習フレームワークにW&Bを統合する方法についての情報を得る。
4. W&B Pythonライブラリ、CLI、そしてクエリ言語の操作に関する技術仕様のための[API Reference guide](../ref/README.md)を参照。

## W&Bはどのように機能しますか？

初めてW&Bを使うユーザーには、以下のセクションをこの順に読むことをお勧めします:

1. W&Bの基本的な計算単位である[Runs](./runs/intro.md)について学ぶ。
2. [Experiments](./track/intro.md)を使って機械学習実験を作成・追跡。
3. データセットとモデルのバージョン管理のための柔軟で軽量なビルディングブロックである[Artifacts](./artifacts/intro.md)を発見。
4. ハイパーパラメータ検索を自動化し、可能なモデルの空間を探索するための[Sweeps](./sweeps/intro.md)を利用。
5. トレーニングからプロダクションまでのモデルライフサイクルを管理する[Model Management](./model_registry/intro.md)。
6. モデルバージョンごとの予測を可視化するための[Data Visualization](./tables/intro.md)ガイド。
7. W&B Runsを整理し、可視化を埋め込み自動化し、発見を記述し、共同作業者と更新情報を共有するための[Reports](./reports/intro.md)。