---
title: Does Launch support parallelization?  How can I limit the resources consumed
  by a job?
menu:
  launch:
    identifier: ja-launch-launch-faq-launch_support_parallelization_limit_resources_consumed_job
    parent: launch-faq
---

Launch は、複数の GPU とノードにわたるジョブのスケーリングをサポートしています。詳細については、[このガイド]({{< relref path="/launch/integration-guides/volcano.md" lang="ja" >}}) を参照してください。

各 Launch エージェントは `max_jobs` パラメータで設定されており、それが同時に実行できるジョブの最大数を決定します。複数のエージェントが適切なローンンチ インフラストラクチャーに接続されている限り、単一のキューを指すことができます。

キューまたはジョブ run レベルで、CPU、GPU、メモリ、その他のリソースに制限を設けることができます。Kubernetes 上でリソース制限付きのキューを設定する方法については、[このガイド]({{< relref path="/launch/set-up-launch/setup-launch-kubernetes.md" lang="ja" >}}) を参照してください。

スイープの場合、同時実行される run の数を制限するために、次のブロックをキュー設定に含めます。

```yaml title="queue config"
  scheduler:
    num_workers: 4
```