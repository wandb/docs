---
title: Kubernetes でエージェントにはどのような権限が必要ですか？
menu:
  launch:
    identifier: ja-launch-launch-faq-permissions_agent_require_kubernetes
    parent: launch-faq
---

Kubernetesマニフェストは、`wandb` ネームスペースで `wandb-launch-agent` という名前のロールを作成します。このロールは、エージェントが `wandb` ネームスペースでポッド、configmaps、secretsを作成し、ポッドのログに アクセス することを可能にします。`wandb-cluster-role` は、エージェントがポッドを作成し、ポッドのログに アクセス し、secrets、ジョブを作成し、指定されたネームスペース全体でジョブのステータスを確認できるようにします。