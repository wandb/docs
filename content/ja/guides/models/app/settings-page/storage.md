---
title: Manage storage
description: W&B のデータストレージを管理する方法。
menu:
  default:
    identifier: ja-guides-models-app-settings-page-storage
    parent: settings
weight: 60
---

ストレージ制限に近づいている、または超過している場合は、 データを管理するために複数の方法があります。最適な方法は、アカウントの種類と現在の プロジェクト の設定によって異なります。

## ストレージ消費量の管理
W&B は、ストレージ消費量を最適化するためのさまざまな メソッド を提供しています。

- [参照 アーティファクト ]({{< relref path="/guides/core/artifacts/track-external-files.md" lang="ja" >}}) を使用して、W&B ストレージにアップロードする代わりに、W&B システムの外部に保存されたファイルを追跡します。
- ストレージに [外部 クラウド ストレージ バケット ]({{< relref path="teams.md" lang="ja" >}}) を使用します。 *(Enterprise のみ)*

## データの削除
ストレージ制限内に収まるように、データを削除することもできます。これを行うには、いくつかの方法があります。

- アプリ UI で対話的にデータを削除します。
- アーティファクト に対して [TTL ポリシーを設定]({{< relref path="/guides/core/artifacts/manage-data/ttl.md" lang="ja" >}}) して、自動的に削除されるようにします。
