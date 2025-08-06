---
title: 誰が私の Artifacts にアクセスできますか？
menu:
  support:
    identifier: ja-support-kb-articles-access_artifacts
support:
- アーティファクト
toc_hide: true
type: docs
url: /support/:filename
---

Artifacts は親プロジェクトからアクセス権限を継承します。

* プライベートプロジェクトの場合、Artifacts へアクセスできるのはチームメンバーのみです。
* パブリックプロジェクトの場合、すべてのユーザーが Artifacts を閲覧できますが、作成や編集ができるのはチームメンバーだけです。
* オープンプロジェクトの場合、すべてのユーザーが Artifacts の閲覧と編集を行えます。

## Artifacts ワークフロー

このセクションでは、Artifacts の管理や編集のためのワークフローについて説明します。多くのワークフローでは [W&B API]({{< relref path="/guides/models/track/public-api-guide.md" lang="ja" >}}) を利用します。これは [クライアントライブラリ]({{< relref path="/ref/python/" lang="ja" >}}) の一部であり、W&B に保存されたデータへのアクセスを提供します。