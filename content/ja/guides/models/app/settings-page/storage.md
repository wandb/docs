---
title: ストレージを管理する
description: W&B データ ストレージを管理する方法。
menu:
  default:
    identifier: ja-guides-models-app-settings-page-storage
    parent: settings
weight: 60
---

If you are approaching or exceeding your storage limit, there are multiple paths forward to manage your data. The path that's best for you will depend on your account type and your current project setup.

## ストレージ消費の管理
W&B は、ストレージ消費を最適化するためのさまざまなメソッドを提供しています:

- [reference Artifacts]({{< relref path="/guides/core/artifacts/track-external-files.md" lang="ja" >}}) を使用して、W&B システム外に保存されたファイルを追跡し、それらを W&B ストレージにアップロードする代わりに使用してください。
- ストレージには[外部クラウドストレージバケット]({{< relref path="teams.md" lang="ja" >}})を使用します。 *(エンタープライズのみ)*

## データの削除
ストレージ制限以下に留めるためにデータを削除することも選択できます。これを行う方法はいくつかあります:

- アプリの UI を使って対話的にデータを削除します。
- Artifacts に [TTL ポリシーを設定]({{< relref path="/guides/core/artifacts/manage-data/ttl.md" lang="ja" >}}) し、自動的に削除されるようにします。