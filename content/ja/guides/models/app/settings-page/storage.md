---
title: Manage storage
description: W&B の データストレージを管理する方法。
menu:
  default:
    identifier: ja-guides-models-app-settings-page-storage
    parent: settings
weight: 60
---

ストレージ制限に近づいている、または超過している場合、データを管理するための複数の方法があります。最適な方法は、アカウントの種類と現在のプロジェクトの設定によって異なります。

## ストレージ消費量の管理
W&B は、ストレージ消費量を最適化するためのさまざまなメソッドを提供しています。

- [参照 Artifacts ]({{< relref path="/guides/core/artifacts/track-external-files.md" lang="ja" >}}) を使用して、W&B ストレージにアップロードする代わりに、W&B システムの外部に保存されたファイルを追跡します。
- ストレージに[外部クラウドストレージバケット]({{< relref path="teams.md" lang="ja" >}})を使用します。 （エンタープライズのみ）

## データを削除する
ストレージ制限内に収まるように、データを削除することもできます。これを行うには、いくつかの方法があります。

- アプリ UI を使用してインタラクティブにデータを削除します。
- Artifacts に [TTL ポリシーを設定]({{< relref path="/guides/core/artifacts/manage-data/ttl.md" lang="ja" >}}) して、自動的に削除されるようにします。
