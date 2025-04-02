---
title: W&B Core
menu:
  default:
    identifier: ja-guides-core-_index
no_list: true
weight: 5
---

W&B Core は、[W&B Models]({{< relref path="/guides/models/" lang="ja" >}}) と [W&B Weave]({{< relref path="/guides/weave/" lang="ja" >}}) をサポートする基盤となる フレームワーク であり、それ自体が [W&B Platform]({{< relref path="/guides/hosting/" lang="ja" >}}) によってサポートされています。

{{< img src="/images/general/core.png" alt="" >}}

W&B Core は、ML ライフサイクル全体にわたる機能を提供します。W&B Core を使用すると、次のことができます。

- 簡単な監査と 再現性 のために、完全な リネージ 追跡を使用して ML [ バージョン を管理]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) パイプライン を行います。
- [インタラクティブで設定可能な 可視化 ]({{< relref path="/guides/models/tables/" lang="ja" >}}) を使用して、 データ と メトリクス を探索および評価します。
- 技術者以外の 関係者 が容易に理解できる、理解しやすいビジュアル形式で ライブ レポート を生成することにより、組織全体の インサイト を [文書化し、共有]({{< relref path="/guides/core/reports/" lang="ja" >}}) します。
- カスタム ニーズ に対応する [ データ の 可視化 をクエリして作成]({{< relref path="/guides/models/app/features/panels/query-panels/" lang="ja" >}}) します。
- [ シークレット を使用して機密性の高い文字列を保護]({{< relref path="/guides/core/secrets.md" lang="ja" >}}) します。
- [model CI/CD]({{< relref path="/guides/core/automations/" lang="ja" >}}) の キー ワークフロー をトリガーする オートメーション を構成します。
