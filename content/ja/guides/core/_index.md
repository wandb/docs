---
title: W&B コア
menu:
  default:
    identifier: ja-guides-core-_index
no_list: true
weight: 5
---

W&B Core は [W&B Models]({{< relref path="/guides/models/" lang="ja" >}}) と [W&B Weave]({{< relref path="/guides/weave/" lang="ja" >}}) をサポートする基盤フレームワークであり、[W&B Platform]({{< relref path="/guides/hosting/" lang="ja" >}}) によってサポートされています。

{{< img src="/images/general/core.png" alt="" >}}

W&B Core は、ML ライフサイクル全体にわたる機能を提供します。W&B Core を使用すると、次のことができます:

- 完全なリネージトレースを使用して [ML パイプラインのバージョン管理と管理]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) を行い、簡単に監査と再現性を確保します。
- [インタラクティブで設定可能な可視化]({{< relref path="/guides/models/tables/" lang="ja" >}})を使用して、データとメトリクスを探索および評価します。
- [レポートを生成し、組織全体で洞察を文書化し共有します]({{< relref path="/guides/core/reports/" lang="ja" >}})。非技術系の利害関係者にも理解しやすい統計化された形式でライブレポートを生成することで達成します。
- [カスタムニーズに合わせたデータのクエリと可視化を作成します]({{< relref path="/guides/models/app/features/panels/query-panels/" lang="ja" >}})。
- [秘密情報をシークレットで保護します]({{< relref path="/guides/core/secrets.md" lang="ja" >}})。
- [モデル CI/CD]({{< relref path="/guides/core/automations/" lang="ja" >}}) のためのキーとなるワークフローをトリガーするオートメーションを設定します。