---
title: Manage storage
description: W&B データ ストレージを管理する方法。
menu:
  default:
    identifier: ja-guides-models-app-settings-page-storage
    parent: settings
weight: 60
---

ストレージ上限に近づいている場合、または超過している場合には、データを管理するための複数の方法があります。最適な方法は、アカウントタイプと現在のプロジェクト設定によって異なります。

## ストレージ消費の管理
W&B は、ストレージ消費を最適化するためのさまざまなメソッドを提供しています:

- W&B ストレージにアップロードするのではなく、W&B システム外に保存されたファイルを追跡するために[参照 Artifacts]({{< relref path="/guides/core/artifacts/track-external-files.md" lang="ja" >}}) を使用します。
- [外部クラウドストレージバケット]({{< relref path="teams.md" lang="ja" >}}) をストレージに使用します。（*エンタープライズのみ*）

## データの削除
ストレージ上限を下回るように、データを削除することも選択できます。これを行う方法は次の通りです:

- アプリ UI を使用して対話的にデータを削除します。
- [TTL ポリシーを設定する]({{< relref path="/guides/core/artifacts/manage-data/ttl.md" lang="ja" >}}) ことで、Artifacts が自動的に削除されるようにします。