---
title: ストレージの管理
description: W&B のデータストレージを管理する方法
menu:
  default:
    identifier: ja-guides-models-app-settings-page-storage
    parent: settings
weight: 60
---

ストレージの上限に近づいている、または超過している場合、データを管理するための複数の選択肢があります。どの方法が最適かは、ご利用のアカウントの種類や現在のプロジェクト構成によって異なります。

## ストレージ消費量の管理
W&B ではストレージ消費量を最適化するためのさまざまな方法を提供しています。

- [参照 Artifacts]({{< relref path="/guides/core/artifacts/track-external-files.md" lang="ja" >}}) を利用して、W&B システム外で保存されたファイルを追跡し、それらを W&B ストレージにアップロードせずに管理できます。
- [外部クラウドストレージバケット]({{< relref path="teams.md" lang="ja" >}}) をストレージとして利用することも可能です。 *(Enterprise のみ)*

## データの削除
ストレージ上限内に収めるために、データを削除することもできます。いくつかの方法があります。

- アプリの UI を使ってデータをインタラクティブに削除する。
- [TTL ポリシーを設定]({{< relref path="/guides/core/artifacts/manage-data/ttl.md" lang="ja" >}})して、Artifacts を自動的に削除できるようにする。