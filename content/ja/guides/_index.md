---
title: ガイド
description: W&B とは何かの概要に加えて、初めてのユーザーの場合の開始方法へのリンクを提供します。
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

Weights & Biases (W&B) は、AI 開発者向けのプラットフォームで、モデルのトレーニング、ファインチューニング、および基盤モデルの活用のためのツールを提供しています。

{{< img src="/images/general/architecture.png" alt="" >}}

W&B は、3つの主要なコンポーネントで構成されています：[Models]({{< relref path="/guides/models.md" lang="ja" >}})、[Weave](https://wandb.github.io/weave/)、および [Core]({{< relref path="/guides/core/" lang="ja" >}}):

**[W&B Models]({{< relref path="/guides/models/" lang="ja" >}})** は、機械学習エンジニアがモデルをトレーニングおよびファインチューニングするための軽量で相互運用可能なツールセットです。
- [Experiments]({{< relref path="/guides/models/track/" lang="ja" >}}): 機械学習実験管理
- [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}): ハイパーパラメータチューニングとモデル最適化
- [Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}): あなたの ML モデルとデータセットを公開して共有

**[W&B Weave]({{< relref path="/guides/weave/" lang="ja" >}})** は、LLM アプリケーションをトラッキングおよび評価するための軽量ツールキットです。

**[W&B Core]({{< relref path="/guides/core/" lang="ja" >}})** は、データとモデルをトラッキングおよび可視化し、結果を伝えるための強力な構成要素セットです。
- [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}): アセットのバージョン管理とリネージのトラック
- [Tables]({{< relref path="/guides/models/tables/" lang="ja" >}}): 表形式データの可視化とクエリ
- [Reports]({{< relref path="/guides/core/reports/" lang="ja" >}}): 発見を文書化し、協力

## W&B はどのように機能しますか？

W&B を初めて使用するユーザーで、機械学習モデルと実験のトレーニング、トラッキング、可視化に興味がある場合、次のセクションをこの順番で読んでください。

1. W&B の基本的な計算単位である [runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) について学びます。
2. [Experiments]({{< relref path="/guides/models/track/" lang="ja" >}}) を使用して機械学習実験を作成し、トラッキングします。
3. データセットとモデルのバージョン管理のための W&B の柔軟で軽量な構成要素を [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) で発見します。
4. ハイパーパラメータ検索を自動化し、可能性のあるモデルの空間を [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) で探索します。
5. モデルのライフサイクルをトレーニングからプロダクションまで管理する [Registry]({{< relref path="/guides/core/registry/" lang="ja" >}})。
6. [Data Visualization]({{< relref path="/guides/models/tables/" lang="ja" >}}) ガイドでモデルバージョン間の予測を可視化します。
7. runs を整理し、可視化を埋め込み、自動化し、学びを説明し、共著者と更新を共有するために [Reports]({{< relref path="/guides/core/reports/" lang="ja" >}}) を使用します。

<iframe width="100%" height="330" src="https://www.youtube.com/embed/tHAFujRhZLA" title="Weights &amp; Biases End-to-End Demo" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## W&B の初めてのユーザーですか？

W&B のインストール方法と W&B をコードに追加する方法を学ぶために、[quickstart]({{< relref path="/guides/quickstart/" lang="ja" >}}) を試してみてください。