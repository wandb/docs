---
title: Kubernetes でエージェントに必要な権限は何ですか？
menu:
  launch:
    identifier: ja-launch-launch-faq-permissions_agent_require_kubernetes
    parent: launch-faq
---

次の Kubernetes マニフェストは、`wandb` 名前空間に `wandb-launch-agent` というロールを作成します。このロールは、エージェントが `wandb` 名前空間内で Pod、ConfigMap、Secret を作成し、Pod のログにアクセスできるようにします。`wandb-cluster-role` は、エージェントが指定した任意の名前空間をまたいで Pod を作成し、Pod のログにアクセスし、Secret や Job を作成し、Job のステータスを確認できるようにします。