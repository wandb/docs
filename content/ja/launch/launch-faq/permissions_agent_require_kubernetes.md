---
title: Kubernetes でエージェントが必要とする権限は何ですか？
menu:
  launch:
    identifier: ja-launch-launch-faq-permissions_agent_require_kubernetes
    parent: launch-faq
---

以下の Kubernetes マニフェストは、`wandb` ネームスペースに `wandb-launch-agent` という名前のロールを作成します。このロールにより、エージェントは `wandb` ネームスペース内で Pod、ConfigMap、Secret の作成や Pod のログ へのアクセスが可能となります。`wandb-cluster-role` は、エージェントが指定した任意のネームスペースで Pod を作成し、Pod のログ にアクセスし、Secret や Job の作成、Job のステータス確認を行えるようにします。