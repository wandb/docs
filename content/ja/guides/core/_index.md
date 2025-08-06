---
title: W&B コア
menu:
  default:
    identifier: core
weight: 5
no_list: true
---

W&B Core は、[W&B Models]({{< relref "/guides/models/" >}}) や [W&B Weave]({{< relref "/guides/weave/" >}}) を支える基盤フレームワークであり、自身も [W&B Platform]({{< relref "/guides/hosting/" >}}) によってサポートされています。

{{< img src="/images/general/core.png" alt="W&B Core framework diagram" >}}

W&B Core は、ML ライフサイクル全体をカバーする機能を提供します。W&B Core を使うことで、次のことが可能です：

- [ML パイプラインをバージョン管理し、フルリネージで追跡]({{< relref "/guides/core/artifacts/" >}}) することで、監査や再現性を簡単に実現できます。
- [インタラクティブで柔軟に設定できる可視化機能]({{< relref "/guides/models/tables/" >}}) を活用して、データやメトリクスを探索・評価できます。
- [ライブレポートの生成・共有]({{< relref "/guides/core/reports/" >}}) により、非技術系の関係者にも分かりやすいビジュアルな形で、組織全体に知見を届けられます。
- [カスタムニーズに合わせてデータのクエリ・可視化作成]({{< relref "/guides/models/app/features/panels/query-panels/" >}}) が行えます。
- [シークレットを使って機密情報の保護]({{< relref "/guides/core/secrets.md" >}}) を行えます。
- [モデル CI/CD のための重要なワークフローを自動化]({{< relref "/guides/core/automations/" >}}) するオートメーションの設定が可能です。