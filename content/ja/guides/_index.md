---
title: Guides
description: W&B の概要と、初めての ユーザー 向けの開始方法に関するリンクを紹介します。
cascade:
  type: docs
menu:
  default:
    identifier: ja-guides-_index
    weight: 1
no_list: true
type: docs
---

## W&Bとは？

Weights & Biases (W&B) は、AI 開発者向けプラットフォームであり、モデルのトレーニング、モデルの微調整、基盤モデルの活用を支援するツールを提供します。

{{< img src="/images/general/architecture.png" alt="" >}}

W&B は、[Models]({{< relref path="/guides/models.md" lang="ja" >}}), [Weave](https://wandb.github.io/weave/), そして [Core]({{< relref path="/guides/core/" lang="ja" >}}) の3つの主要なコンポーネントで構成されています。

**[W&B Models]({{< relref path="/guides/models/" lang="ja" >}})** は、機械学習エンジニアがモデルのトレーニングと微調整を行うための、軽量で相互運用可能なツールセットです。
- [Experiments]({{< relref path="/guides/models/track/" lang="ja" >}}): 機械学習 の 実験管理
- [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}): ハイパーパラメータチューニング と モデル最適化
- [Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}): ML モデル と データセット を公開して共有

**[W&B Weave]({{< relref path="/guides/weave/" lang="ja" >}})** は、LLM アプリケーション を追跡および評価するための軽量なツールキットです。

**[W&B Core]({{< relref path="/guides/core/" lang="ja" >}})** は、データとモデルの追跡と可視化、結果の伝達を行うための強力な構成要素のセットです。
- [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}): アセット の バージョン管理 と リネージ の追跡
- [Tables]({{< relref path="/guides/models/tables/" lang="ja" >}}): テーブル データの可視化とクエリ
- [Reports]({{< relref path="/guides/core/reports/" lang="ja" >}}): 発見事項の文書化と共同作業

## W&B の仕組み

W&B を初めて使用するユーザーで、機械学習モデルと実験のトレーニング、追跡、可視化に関心がある場合は、次のセクションをこの順序でお読みください。

1. W&B の基本的な計算単位である [runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) について学びます。
2. [Experiments]({{< relref path="/guides/models/track/" lang="ja" >}}) を使用して、機械学習の実験を作成および追跡します。
3. [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) を使用して、データセット と モデル の バージョン管理を行うための、W&B の柔軟で軽量な構成要素を見つけます。
4. [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) を使用して、ハイパーパラメーター 探索を自動化し、可能なモデルの空間を探索します。
5. [Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) を使用して、トレーニング から プロダクション までの モデル ライフサイクルを管理します。
6. [Data Visualization]({{< relref path="/guides/models/tables/" lang="ja" >}}) ガイドで、モデル バージョン全体の予測を可視化します。
7. [Reports]({{< relref path="/guides/core/reports/" lang="ja" >}}) を使用して、runs の整理、可視化の埋め込みと自動化、発見事項の記述、および コラボレーター との更新の共有を行います。

<iframe width="100%" height="330" src="https://www.youtube.com/embed/tHAFujRhZLA" title="Weights &amp; Biases End-to-End Demo" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## W&B を初めて使用しますか？

[quickstart]({{< relref path="/guides/quickstart/" lang="ja" >}}) を試して、W&B のインストール方法と、W&B を コード に追加する方法を学んでください。
