---
title: 誰が自分の Artifacts にアクセスできますか？
menu:
  support:
    identifier: ja-support-kb-articles-access_artifacts
support:
- Artifacts
toc_hide: true
type: docs
url: /support/:filename
---

Artifacts は親プロジェクトのアクセス権限を継承します:

* 非公開のプロジェクトでは、チームメンバーのみが Artifacts にアクセスできます。
* 公開プロジェクトでは、すべてのユーザーが Artifacts を閲覧でき、作成や変更はチームメンバーのみが行えます。
* オープンプロジェクトでは、すべてのユーザーが Artifacts を読み書きできます。

## Artifacts ワークフロー

このセクションでは、Artifacts の管理および編集に関するワークフローを説明します。多くのワークフローは [W&B API]({{< relref path="/guides/models/track/public-api-guide.md" lang="ja" >}}) を利用します。これは [クライアント ライブラリ]({{< relref path="/ref/python/" lang="ja" >}}) の一部で、W&B に保存されたデータへのアクセスを提供します。