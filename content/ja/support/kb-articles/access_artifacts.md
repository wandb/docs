---
title: あなたのアーティファクトへのアクセス権を持っているのは誰ですか？
menu:
  support:
    identifier: ja-support-kb-articles-access_artifacts
support:
- artifacts
toc_hide: true
type: docs
url: /support/:filename
---

Artifacts は、その親プロジェクトからアクセス権限を継承します:

* プライベートプロジェクトでは、アーティファクトにアクセスできるのはチームメンバーのみです。
* パブリックプロジェクトでは、全ユーザーがアーティファクトを読むことができますが、作成または変更することができるのはチームメンバーのみです。
* オープンプロジェクトでは、全ユーザーがアーティファクトを読んだり書いたりできます。

## Artifacts ワークフロー

このセクションでは、Artifacts の管理と編集に関するワークフローを概説します。多くのワークフローは [the W&B API]({{< relref path="/guides/models/track/public-api-guide.md" lang="ja" >}})、および W&B に保存されたデータへのアクセスを提供する[クライアントライブラリ]({{< relref path="/ref/python/" lang="ja" >}})のコンポーネントを利用しています。