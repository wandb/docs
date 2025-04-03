---
title: Who has access to my artifacts?
menu:
  support:
    identifier: ja-support-kb-articles-access_artifacts
support:
- artifacts
toc_hide: true
type: docs
url: /support/:filename
---

Artifacts は、親の project からアクセス権を継承します。

* プライベート project では、チームメンバーのみが Artifacts にアクセスできます。
* パブリック project では、すべてのユーザーが Artifacts を読み取ることができ、チームメンバーのみが Artifacts を作成または変更できます。
* オープン project では、すべてのユーザーが Artifacts を読み書きできます。

## Artifacts の ワークフロー

このセクションでは、Artifacts の管理および編集に関するワークフローの概要を説明します。多くのワークフローは、[W&B API]({{< relref path="/guides/models/track/public-api-guide.md" lang="ja" >}})を利用します。これは、W&B に保存された data へのアクセスを提供する[クライアント library]({{< relref path="/ref/python/" lang="ja" >}})のコンポーネントです。
