---
title: ガイド
description: W&B の概要と開始方法
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

W&B は AI 開発者向けプラットフォームであり、モデルのトレーニング、ファインチューニング、基盤モデルの活用のための各種ツールを提供しています。

{{< img src="/images/general/architecture.png" alt="W&B platform architecture diagram" >}}

W&B は、主に３つの主要コンポーネントから構成されています：[Models]({{< relref path="/guides/models.md" lang="ja" >}})、[Weave](https://wandb.github.io/weave/)、そして [Core]({{< relref path="/guides/core/" lang="ja" >}}) です。

**[W&B Models]({{< relref path="/guides/models/" lang="ja" >}})** は、機械学習エンジニアがモデルをトレーニング・ファインチューニングするための、軽量かつ相互運用可能なツール群です。
- [Experiments]({{< relref path="/guides/models/track/" lang="ja" >}}): 機械学習の実験管理
- [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}): ハイパーパラメータチューニングとモデル最適化
- [Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}): あなたの ML モデルやデータセットを公開・共有

**[W&B Weave]({{< relref path="/guides/weave/" lang="ja" >}})** は、LLMアプリケーションの追跡・評価のための軽量ツールキットです。

**[W&B Core]({{< relref path="/guides/core/" lang="ja" >}})** は、データやモデルの可視化や追跡、結果の共有を支援する強力なビルディングブロック群です。
- [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}): 資産のバージョン管理とリネージ追跡
- [Tables]({{< relref path="/guides/models/tables/" lang="ja" >}}): 表形式データの可視化・クエリ
- [Reports]({{< relref path="/guides/core/reports/" lang="ja" >}}): 新たな発見のドキュメント化と共同作業

{{% alert %}}
直近のリリース情報は [W&B リリースノート]({{< relref path="/ref/release-notes/" lang="ja" >}}) をご参照ください。
{{% /alert %}}

## W&B はどのように動作しますか？

W&B を初めて使う方で、機械学習モデルや実験のトレーニング、管理、可視化に関心がある場合は、以下のセクションを順にお読みください。

1. [runs]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) について学びましょう。これは W&B の基本的な計算単位です。
2. [Experiments]({{< relref path="/guides/models/track/" lang="ja" >}}) を使って機械学習実験を作成・管理します。
3. データセット・モデルのバージョン管理と柔軟な活用を実現する [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) をチェックしましょう。
4. ハイパーパラメータの自動探索やモデル探索には [Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) を活用します。
5. トレーニングからプロダクションまで、モデルのライフサイクルを管理するには [Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) があります。
6. モデルの異なるバージョンでの予測を比較・可視化するには [Data Visualization]({{< relref path="/guides/models/tables/" lang="ja" >}}) ガイドをご覧ください。
7. runs の整理、可視化の自動化や埋め込み、学びのまとめや共有は [Reports]({{< relref path="/guides/core/reports/" lang="ja" >}}) がおすすめです。

<iframe width="100%" height="330" src="https://www.youtube.com/embed/tHAFujRhZLA" title="Weights &amp; Biases End-to-End Demo" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## W&B を初めてご利用ですか？

[クイックスタート]({{< relref path="/guides/quickstart/" lang="ja" >}}) を試して、W&B のインストール方法やコードへの追加方法を学んでみましょう。