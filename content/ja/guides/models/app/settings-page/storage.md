---
title: ストレージの管理
description: W&B のデータ ストレージを管理する方法。
menu:
  default:
    identifier: ja-guides-models-app-settings-page-storage
    parent: settings
weight: 60
---

ストレージ上限に近づいている、または超過している場合、データを管理するための方法が複数あります。最適な方法は、アカウントの種類や現在のプロジェクト構成によって異なります。

## ストレージ使用量の管理
W&B では、ストレージ使用量を最適化するためのさまざまな方法を提供しています:

- [reference Artifacts]({{< relref path="/guides/core/artifacts/track-external-files.md" lang="ja" >}}) を使用して、W&B ストレージにアップロードする代わりに W&B システム外に保存したファイルを追跡します。
- ストレージとして [外部の クラウド ストレージ バケット]({{< relref path="teams.md" lang="ja" >}}) を使用します。*(Enterprise のみ)*

## データの削除
ストレージ上限内に収めるために、データを削除することもできます。これにはいくつかの方法があります:

- アプリの UI から対話的にデータを削除します。
- Artifacts に [TTL ポリシーを設定]({{< relref path="/guides/core/artifacts/manage-data/ttl.md" lang="ja" >}}) し、自動的に削除されるようにします。