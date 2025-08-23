---
title: W&B Models
menu:
  default:
    identifier: ja-guides-models-_index
no_list: true
weight: 3
---

W&B Models は、モデルを整理し、生産性とコラボレーションを向上させ、スケールでプロダクション ML を実現したい機械学習エンジニアのための SoR（System of Record）です。

{{< img src="/images/general/architecture.png" alt="W&B Models architecture diagram" >}}

W&B Models を使うことで、次のことが可能になります：

- すべての [ML 実験]({{< relref path="/guides/models/track/" lang="ja" >}}) をトラッキング・可視化できます。
- [ハイパーパラメーター探索]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) でモデルをスケールして最適化・ファインチューンできます。
- すべてのモデルを [中央ハブとして管理]({{< relref path="/guides/core/registry/" lang="ja" >}}) でき、DevOps やデプロイメントへのシームレスな引き継ぎも可能です。
- [モデル CI/CD]({{< relref path="/guides/core/automations/" lang="ja" >}}) のための主要なワークフローをトリガーするカスタムオートメーションを設定できます。

機械学習エンジニアは、W&B Models を SoR として活用し、実験のトラッキングや可視化、モデルのバージョンやリネージ管理、ハイパーパラメーターの最適化を行っています。