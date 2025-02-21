---
title: Does Launch support parallelization?  How can I limit the resources consumed
  by a job?
menu:
  launch:
    identifier: ja-launch-launch-faq-launch_support_parallelization_limit_resources_consumed_job
    parent: launch-faq
---

Launch は、複数の GPU とノードにまたがってジョブをスケールすることをサポートしています。詳細については、[こちらのガイド]({{< relref path="/launch/integration-guides/volcano.md" lang="ja" >}})を参照してください。

各 Launch エージェントは `max_jobs` パラメータで設定され、これは実行できる同時ジョブの最大数を決定します。複数のエージェントが、適切な起動インフラストラクチャに接続している限り、単一のキューを指すことができます。

リソース構成では、CPU、GPU、メモリ、およびその他のリソースの制限を、キューまたはジョブの run レベルで設定できます。Kubernetes でリソース制限付きのキューをセットアップする方法については、[こちらのガイド]({{< relref path="/launch/set-up-launch/setup-launch-kubernetes.md" lang="ja" >}})を参照してください。

Sweeps の場合、同時 run の数を制限するには、次のブロックをキュー構成に含めます。

```yaml title="queue config"
  scheduler:
    num_workers: 4
```
