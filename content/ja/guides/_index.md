---
title: ガイド
description: W&B の概要と開始方法について説明します。
menu:
  default:
    identifier: guides
    weight: 1
type: docs
cascade:
  type: docs
no_list: true
---

## W&B とは？

W&B は AI 開発者向けプラットフォームであり、モデルのトレーニング、ファインチューニング、ファウンデーションモデルの活用までをカバーする各種ツールを提供します。

{{< img src="/images/general/architecture.png" alt="W&B プラットフォーム アーキテクチャ図" >}}

W&B は大きく 3 つのコンポーネントで構成されています: [Models]({{< relref "/guides/models.md" >}})、[Weave](https://wandb.github.io/weave/)、[Core]({{< relref "/guides/core/" >}}):

**[W&B Models]({{< relref "/guides/models/" >}})** は、機械学習エンジニア向けのモデルのトレーニングやファインチューニングを支援する軽量で相互運用性の高いツール群です。
- [Experiments]({{< relref "/guides/models/track/" >}}): 機械学習実験管理
- [Sweeps]({{< relref "/guides/models/sweeps/" >}}): ハイパーパラメータチューニングやモデル最適化
- [Registry]({{< relref "/guides/core/registry/" >}}): ML モデルやデータセットの公開・共有

**[W&B Weave]({{< relref "/guides/weave/" >}})** は、LLM アプリケーションの追跡や評価を実現する軽量なツールキットです。

**[W&B Core]({{< relref "/guides/core/" >}})** は、データやモデルのトラッキング、可視化、結果の共有に便利な強力なビルディングブロック群です。
- [Artifacts]({{< relref "/guides/core/artifacts/" >}}): 資産のバージョン管理とリネージ追跡
- [Tables]({{< relref "/guides/models/tables/" >}}): テーブルデータの可視化およびクエリ
- [Reports]({{< relref "/guides/core/reports/" >}}): 発見した内容のドキュメント化やコラボレーション

{{% alert %}}
最新リリース情報は [W&B リリースノート]({{< relref "/ref/release-notes/" >}}) をご覧ください。
{{% /alert %}}

## W&B はどのように機能しますか？

W&B を初めて利用する場合、下記の順番でセクションを読むと、モデルや実験のトレーニング、トラッキング、可視化について理解できます。

1. W&B の基本単位である [runs]({{< relref "/guides/models/track/runs/" >}}) について学びます。
2. [Experiments]({{< relref "/guides/models/track/" >}}) で機械学習実験を作成・管理してみましょう。
3. データセットやモデルを柔軟かつ軽量にバージョン管理できる [Artifacts]({{< relref "/guides/core/artifacts/" >}}) について知りましょう。
4. [Sweeps]({{< relref "/guides/models/sweeps/" >}}) を使ってハイパーパラメータ探索やモデル最適化を自動化しましょう。
5. [Registry]({{< relref "/guides/core/registry/" >}}) を活用して、トレーニングからプロダクションまでモデルライフサイクルを管理します。
6. [Data Visualization]({{< relref "/guides/models/tables/" >}}) ガイドでモデルごとの予測を可視化しましょう。
7. [Reports]({{< relref "/guides/core/reports/" >}}) で run を整理し、可視化の埋め込みや自動化、発見内容の記述、コラボレーターとの共有を行いましょう。

<iframe width="100%" height="330" src="https://www.youtube.com/embed/tHAFujRhZLA" title="Weights &amp; Biases End-to-End Demo" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## W&B を初めて使う方へ

[クイックスタート]({{< relref "/guides/quickstart/" >}}) を試して、W&B のインストールや、コードへの組み込み方法を学びましょう。