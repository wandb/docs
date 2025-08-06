---
title: Kubernetes でエージェントに必要な権限は何ですか？
menu:
  launch:
    identifier: permissions_agent_require_kubernetes
    parent: launch-faq
---

以下の Kubernetes マニフェストは、`wandb` ネームスペース内に `wandb-launch-agent` というロールを作成します。このロールはエージェントに、`wandb` ネームスペースで Pod、ConfigMap、Secret の作成や Pod のログへのアクセスを許可します。また、`wandb-cluster-role` により、エージェントは指定された任意のネームスペースで Pod の作成、Pod のログへのアクセス、Secret や Job の作成、さらに Job のステータス確認ができるようになります。