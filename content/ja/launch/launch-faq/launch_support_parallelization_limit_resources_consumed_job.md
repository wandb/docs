---
title: Does Launch support parallelization?  How can I limit the resources consumed
  by a job?
menu:
  launch:
    identifier: ja-launch-launch-faq-launch_support_parallelization_limit_resources_consumed_job
    parent: launch-faq
---

Launch は、複数の GPU とノードにまたがるジョブのスケーリングをサポートしています。詳細については、[こちらのガイド]({{< relref path="/launch/integration-guides/volcano.md" lang="ja" >}})を参照してください。

各 Launch エージェント は、実行できる同時ジョブの最大数を決定する `max_jobs` パラメータ で設定されています。複数の エージェント は、適切な起動 インフラストラクチャ に接続している限り、単一のキューを指すことができます。

リソース設定では、CPU、GPU、メモリ、およびその他のリソースの制限を、キューまたはジョブ run レベルで設定できます。Kubernetes でリソース制限付きのキューを設定する方法については、[こちらのガイド]({{< relref path="/launch/set-up-launch/setup-launch-kubernetes.md" lang="ja" >}})を参照してください。

Sweeps の場合、同時 run の数を制限するには、次のブロックをキュー設定に含めます。

```yaml title="queue config"
  scheduler:
    num_workers: 4
```
