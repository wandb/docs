---
description: >-
  An overview of what is Weights & Biases along with links on how to get started
  if you are a first time user.
slug: /
displayed_sidebar: ja
---

# Weights & Biasesとは？

Weights & Biasesは、開発者がより優れたモデルを迅速に構築するための機械学習プラットフォームです。W&Bの軽量で相互運用可能なツールを使って、実験を迅速に追跡し、データセットのバージョニングと反復を行い、モデルパフォーマンスを評価し、モデルを再現し、結果を可視化し、回帰を見つけて発見事項を同僚と共有します。W&Bを5分でセットアップしてから、機械学習開発フローで迅速に反復します。弊社のデータセットとモデルは信頼できるSoR（記録システム）で追跡およびバージョン管理されるため、安心して作業ができます。

<!-- ![](@site/static/images/general/diagram_2021.png) -->

## W&Bを使うのは初めてですか？
W&Bを初めて使う場合、以下を確認することをお勧めします：

1. 動作中のWeights & Biasesを体験し、[Google Colabを使って導入のサンプルプロジェクトを実行する](http://wandb.me/intro).
1. [クイックスタート](../quickstart.md) を読み、W&Bをコードに追加する方法と場所の概要を理解する。
2. [Weights & Biasesの動作の仕組み](#how-does-weights--biases-work) を読むこのセクションでは、W&Bのビルディングブロックの概要を提供します。
3. [統合ガイド](./integrations/intro.md)と[W&B簡単統合YouTube](https://www.youtube.com/playlist?list=PLD80i8An1OEGDADxOBaH71ZwieZ9nmPGC) プレイリストを探索して、W&Bをお好みの機械学習フレームワークと統合する方法に関する情報を入手します。
4. [APIリファレンスガイド](../ref/README.md) を読み、W&B Pythonライブラリ、CLIおよびWeaveオペレーションに関する技術仕様を理解します。


## Weights & Biasesの動作の仕組み
W&Bを初めて使う場合、以下のセクションをこの順序で読むことをお勧めします：

1. [Runs](./runs/intro.md)の詳細、W&Bの計算基本単位を学びます。
2. [実験](./track/intro.md)で、機械学習実験を作成および追跡します。
3. [アーティファクト](./artifacts/intro.md)で、データセットとモデルバージョン管理用の、W&Bの柔軟かつ軽量のビルディングブロックを発見します。
4. ハイパーパラメーター検索を自動化し、スウィープで可能性のあるモデルの空間を探索します。
6. [モデル管理](./models/intro.md)で、トレーニングからプロダクションまでのモデルライフサイクルを管理します。
7. [データ可視化](./data-vis/intro.md)ガイドで、複数のモデルバージョン間の予測を可視化します。
8. W&B Runを体系化し、可視化の埋め込みと自動化を行い、発見事項を説明し、[レポート](./reports/intro.md)でアップデートを共同作業者と共有します。
