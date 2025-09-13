---
title: W&B コア
menu:
  default:
    identifier: ja-guides-core-_index
no_list: true
weight: 5
---

W&B Core は、[W&B Models]({{< relref path="/guides/models/" lang="ja" >}}) と [W&B Weave]({{< relref path="/guides/weave/" lang="ja" >}}) を支える基盤フレームワークで、その土台は [W&B Platform]({{< relref path="/guides/hosting/" lang="ja" >}}) によって提供されています。

{{< img src="/images/general/core.png" alt="W&B Core のフレームワーク図" >}}

W&B Core は、ML ライフサイクル全体にわたって機能を提供します。W&B Core を使用すると、次のことができます。

- [ML パイプラインをバージョン管理し、管理する]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) ことで、完全なリネージ追跡により監査と再現性を容易にします。
- [インタラクティブで構成可能な可視化]({{< relref path="/guides/models/tables/" lang="ja" >}}) を使用して、データとメトリクスを探索し、評価します。
- 非技術系の関係者にもわかりやすいビジュアル形式のライブ Reports を生成し、組織全体でインサイトを [文書化し、共有します]({{< relref path="/guides/core/reports/" lang="ja" >}})。
- カスタム ニーズに合わせて、[データのクエリと可視化を作成します]({{< relref path="/guides/models/app/features/panels/query-panels/" lang="ja" >}})。
- [シークレットを使用して機密性の高い文字列を保護します]({{< relref path="/guides/core/secrets.md" lang="ja" >}})。
- [モデル CI/CD]({{< relref path="/guides/core/automations/" lang="ja" >}}) のための主要なワークフローをトリガーするオートメーションを構成します。