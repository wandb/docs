---
title: W&B コア
menu:
  default:
    identifier: ja-guides-core-_index
no_list: true
weight: 5
---

W&B Core は、[W&B Models]({{< relref path="/guides/models/" lang="ja" >}}) と [W&B Weave]({{< relref path="/guides/weave/" lang="ja" >}}) を支える基盤フレームワークであり、自身も [W&B Platform]({{< relref path="/guides/hosting/" lang="ja" >}}) によってサポートされています。

{{< img src="/images/general/core.png" alt="W&B Core framework diagram" >}}

W&B Core は、ML ライフサイクル全体にわたる機能を提供します。W&B Core を使うことで、以下が可能です。

- [ML パイプラインをバージョン管理し、リネージトレースで監査や再現性を容易にする]({{< relref path="/guides/core/artifacts/" lang="ja" >}})
- [対話的かつ柔軟に設定可能な可視化]({{< relref path="/guides/models/tables/" lang="ja" >}}) を使って、データやメトリクスを探索・評価する
- [ライブレポートを生成し、組織全体へ知見をドキュメント・共有する]({{< relref path="/guides/core/reports/" lang="ja" >}})。技術的知識がなくても理解しやすい、ビジュアル形式でアウトプットできます
- [カスタムニーズに応じてデータをクエリし、可視化を作成する]({{< relref path="/guides/models/app/features/panels/query-panels/" lang="ja" >}})
- [シークレットを活用して機密情報を保護する]({{< relref path="/guides/core/secrets.md" lang="ja" >}})
- [model CI/CD 用の主要なワークフローを自動で実行する]({{< relref path="/guides/core/automations/" lang="ja" >}})オートメーションの設定