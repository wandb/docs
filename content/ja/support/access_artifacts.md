---
title: Who has access to my artifacts?
menu:
  support:
    identifier: ja-support-access_artifacts
tags:
- artifacts
toc_hide: true
type: docs
---

Artifacts は親プロジェクトからアクセス権を継承します:

* プライベートプロジェクトでは、アーティファクトにアクセスできるのはチームメンバーのみです。
* パブリックプロジェクトでは、すべてのユーザーがアーティファクトを読むことができ、作成や変更はチームメンバーのみが可能です。
* オープンプロジェクトでは、すべてのユーザーがアーティファクトを読み書きすることができます。

## Artifacts ワークフロー

このセクションでは、Artifacts の管理と編集のワークフローを示します。多くのワークフローは [the W&B API]({{< relref path="/guides/models/track/public-api-guide.md" lang="ja" >}})、および W&B に保存されたデータへアクセスするための [the client library]({{< relref path="/ref/python/" lang="ja" >}}) を利用します。