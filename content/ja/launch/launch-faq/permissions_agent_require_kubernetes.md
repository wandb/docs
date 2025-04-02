---
title: What permissions does the agent require in Kubernetes?
menu:
  launch:
    identifier: ja-launch-launch-faq-permissions_agent_require_kubernetes
    parent: launch-faq
---

以下の Kubernetes マニフェストは、`wandb` 名前空間に `wandb-launch-agent` という名前のロールを作成します。このロールにより、エージェントは `wandb` 名前空間に Pod、configmap、secret を作成し、Pod の ログ に アクセス できるようになります。`wandb-cluster-role` を使用すると、エージェントは Pod の作成、Pod の ログ への アクセス 、secret、ジョブの作成、および指定された名前空間全体のジョブステータスの確認を行うことができます。
