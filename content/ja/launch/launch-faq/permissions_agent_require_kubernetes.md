---
title: What permissions does the agent require in Kubernetes?
menu:
  launch:
    identifier: ja-launch-launch-faq-permissions_agent_require_kubernetes
    parent: launch-faq
---

以下の Kubernetes マニフェストは、`wandb` namespace に `wandb-launch-agent` という名前のロールを作成します。このロールにより、エージェントは pods、configmaps、secrets を作成し、`wandb` namespace 内の pod の ログ にアクセスできます。`wandb-cluster-role` を使用すると、エージェントは pods の作成、pod の ログ へのアクセス、secrets、jobs の作成、および指定された namespace 全体でのジョブステータスの確認を行うことができます。
