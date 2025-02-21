---
title: What permissions does the agent require in Kubernetes?
menu:
  launch:
    identifier: ja-launch-launch-faq-permissions_agent_require_kubernetes
    parent: launch-faq
---

Kubernetes マニフェストは、`wandb` ネームスペース内に `wandb-launch-agent` という名前のロールを作成します。このロールは、エージェントが `wandb` ネームスペース内でポッド、configmaps、シークレットを作成し、ポッドログに アクセス することを許可します。`wandb-cluster-role` は、エージェントが指定された任意のネームスペースでポッドを作成し、ポッドログに アクセス し、シークレット、ジョブを作成し、ジョブのステータスを確認できるようにします。