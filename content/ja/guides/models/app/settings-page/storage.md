---
title: ストレージの管理
description: W&B のデータストレージを管理する方法
menu:
  default:
    identifier: app_storage
    parent: settings
weight: 60
---

ストレージの上限に近づいている、または超えている場合、データを管理するための複数の方法があります。どの方法が最適かは、アカウントの種類や現在のプロジェクトの設定によって異なります。

## ストレージ消費量の管理
W&B ではストレージ消費を最適化するためのさまざまな方法を提供しています。

- [reference artifacts]({{< relref "/guides/core/artifacts/track-external-files.md" >}}) を使って、W&B システム外に保存されたファイルをトラッキングし、W&B ストレージにアップロードしないようにする
- [外部クラウドストレージバケット]({{< relref "teams.md" >}}) をストレージとして利用する（*Enterpriseプランのみ*）

## データの削除
ストレージ上限を下回るために、データを削除することも可能です。主に次の方法があります。

- アプリ UI からデータをインタラクティブに削除する
- Artifacts に [TTL ポリシー]({{< relref "/guides/core/artifacts/manage-data/ttl.md" >}}) を設定し、自動で削除されるようにする