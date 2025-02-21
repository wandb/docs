---
title: W&B Core
menu:
  default:
    identifier: ja-guides-core-_index
no_list: true
weight: 5
---

W&B Core は [W&B Models]({{< relref path="/guides/models/" lang="ja" >}}) と [W&B Weave]({{< relref path="/guides/weave/" lang="ja" >}}) をサポートする基盤フレームワークであり、[W&B Platform]({{< relref path="/guides/hosting/" lang="ja" >}}) によってサポートされています。

{{< img src="/images/general/core.png" alt="" >}}

W&B Core は、ML ライフサイクル全体にわたる機能を提供します。W&B Core を使用すると、次のことができます。

- 簡単な監査と再現性のために、完全なリネージトレーシングを使用して [Version and manage ML]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) パイプラインを管理する。
- [対話型で構成可能な可視化]({{< relref path="/guides/core/tables/" lang="ja" >}}) を使用して、データとメトリクスを探索および評価する。
- テクニカルでない関係者が簡単に理解できる視覚的形式のライブレポートを生成することにより、[組織全体での洞察のドキュメント化と共有]({{< relref path="/guides/core/reports/" lang="ja" >}}) を行う。
- 独自のニーズに応じたデータの [問い合わせと可視化の作成]({{< relref path="/guides/models/app/features/panels/query-panels/" lang="ja" >}}) を行う。