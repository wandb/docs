---
title: W&B Models
menu:
  default:
    identifier: models
weight: 3
no_list: true
---

W&B Models は、モデルの整理、生産性やコラボレーションの向上、そして大規模なプロダクション級機械学習の実現を目指す ML エンジニア向けのシステム・オブ・レコードです。

{{< img src="/images/general/architecture.png" alt="W&B Models アーキテクチャ図" >}}

W&B Models でできること:

- すべての [ML experiments]({{< relref "/guides/models/track/" >}}) をトラッキングし、可視化できます。
- [hyperparameter sweeps]({{< relref "/guides/models/sweeps/" >}}) を使って、モデルを大規模に最適化・ファインチューンできます。
- [すべてのモデルを集中管理するハブを維持]({{< relref "/guides/core/registry/" >}}) でき、devops やデプロイメントへのシームレスな引き渡しが可能です。
- [model CI/CD]({{< relref "/guides/core/automations/" >}}) 用の主要なワークフローを自動でトリガーするカスタムオートメーションを設定できます。

機械学習エンジニアは、W&B Models を ML のシステム・オブ・レコードとして活用し、実験のトラッキングと可視化、モデルのバージョンやリネージの管理、ハイパーパラメーターの最適化を行っています。