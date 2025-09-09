---
title: ガイド
description: W&B の概要と始め方。
cascade:
  type: docs
menu:
  default:
    identifier: ja-guides-_index
    weight: 1
no_list: true
type: docs
---

## W&B とは？

W&B は、モデルのトレーニング、ファインチューニング、基盤モデルの活用を支援するツールを提供する AI 開発者プラットフォームです。

{{< img src="/images/general/architecture.png" alt="W&B プラットフォームのアーキテクチャー図" >}}

W&B は、主要な3つのコンポーネントである [Models]({{< relref path="/guides/models.md" lang="ja" >}})、[Weave](https://wandb.github.io/weave/)、[Core]({{< relref path="/guides/core/" lang="ja" >}}) で構成されています。

**[W&B Models]({{< relref path="/guides/models/" lang="ja" >}})** は、モデルのトレーニングとファインチューニングを行う機械学習エンジニア向けの、軽量で相互運用可能なツールセットです。
- [Experiments]({{< relref path="/guides/models/track/" lang="ja" >}}): 機械学習の実験管理
- [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}): ハイパーパラメータチューニングとモデルの最適化
- [Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}): 機械学習モデルとデータセットの公開および共有

**[W&B Weave]({{< relref path="/guides/weave/" lang="ja" >}})** は、LLM アプリケーションを追跡および評価するための軽量ツールキットです。

**[W&B Core]({{< relref path="/guides/core/" lang="ja" >}})** は、データとモデルの追跡・可視化、および結果の共有を可能にする強力な構成要素です。
- [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}): アセットのバージョン管理とリネージの追跡
- [Tables]({{< relref path="/guides/models/tables/" lang="ja" >}}): 表形式データの可視化とクエリ
- [Reports]({{< relref path="/guides/core/reports/" lang="ja" >}}): 発見事項の文書化と共同作業

**[W&B Inference]({{< relref path="/guides/inference/" lang="ja" >}})** は、W&B Weave と OpenAI 互換の API を介してオープンソースの基盤モデルにアクセスするためのツールセットです。

{{% alert %}}
[W&B リリースノート]({{< relref path="/ref/release-notes/" lang="ja" >}}) で最新リリースについて学びましょう。
{{% /alert %}}

## W&B はどのように機能しますか？

W&B を初めて使用する方で、機械学習モデルや実験のトレーニング、追跡、可視化に興味がある場合は、以下のセクションをこの順序でお読みください。

1. W&B の基本的な計算単位である [runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) について学びます。
2. [Experiments]({{< relref path="/guides/models/track/" lang="ja" >}}) を使用して、機械学習実験を作成および追跡します。
3. [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) で、データセットとモデルのバージョン管理のための W&B の柔軟で軽量な構成要素を発見します。
4. [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) で、ハイパーパラメーター探索を自動化し、可能なモデルの空間を探索します。
5. [Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) で、トレーニングからプロダクションまでのモデルライフサイクルを管理します。
6. [Data Visualization]({{< relref path="/guides/models/tables/" lang="ja" >}}) ガイドを使用して、モデルバージョン間の予測を可視化します。
7. [Reports]({{< relref path="/guides/core/reports/" lang="ja" >}}) を使用して、Runs を整理し、可視化を埋め込み・自動化し、学びを記述し、共同作業者と更新を共有します。

<iframe width="100%" height="330" src="https://www.youtube.com/embed/tHAFujRhZLA" title="Weights &amp; Biases エンドツーエンド デモ" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## W&B を初めてご利用ですか？

[クイックスタート]({{< relref path="/guides/quickstart/" lang="ja" >}}) を試して、W&B のインストール方法とコードへの追加方法を学びましょう。