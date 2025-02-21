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

Artifacts は、親 プロジェクト からアクセス権を継承します。

* プライベート プロジェクト では、 チームメンバー のみ が Artifacts にアクセスできます。
* パブリック プロジェクト では、すべての ユーザー が Artifacts を読み取ることができ、 チームメンバー のみ が Artifacts を作成または変更できます。
* オープン プロジェクト では、すべての ユーザー が Artifacts を読み書きできます。

## Artifacts の ワークフロー

このセクションでは、 Artifacts の管理および編集の ワークフロー について概説します。多くの ワークフロー は、W&B に保存された データ への アクセス を提供する [クライアント ライブラリ]({{< relref path="/ref/python/" lang="ja" >}}) のコンポーネントである [W&B API]({{< relref path="/guides/models/track/public-api-guide.md" lang="ja" >}}) を利用しています。
