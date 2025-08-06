---
title: 誰が私のアーティファクトにアクセスできますか？
url: /support/:filename
toc_hide: true
type: docs
support:
- アーティファクト
---

Artifacts は親プロジェクトからアクセス権限を継承します。

* プライベートプロジェクトでは、チームメンバーのみが Artifacts にアクセスできます。
* パブリックプロジェクトでは、すべてのユーザーが Artifacts を閲覧でき、作成や編集はチームメンバーのみ可能です。
* オープンプロジェクトでは、すべてのユーザーが Artifacts を閲覧および編集できます。

## Artifacts のワークフロー

このセクションでは、Artifacts の管理および編集のワークフローについて説明します。多くのワークフローでは [the W&B API]({{< relref "/guides/models/track/public-api-guide.md" >}}) を利用します。これは、W&B に保存されているデータへアクセスできる [the client library]({{< relref "/ref/python/" >}}) のコンポーネントのひとつです。