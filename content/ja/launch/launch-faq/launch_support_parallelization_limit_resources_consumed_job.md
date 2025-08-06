---
title: Launch は並列処理に対応していますか？ また、ジョブが消費するリソースを制限するにはどうすればよいですか？
menu:
  launch:
    identifier: launch_support_parallelization_limit_resources_consumed_job
    parent: launch-faq
---

Launch は複数の GPU やノードにまたがるジョブのスケーリングをサポートしています。詳細については、[Volcano インテグレーションガイド]({{< relref "/launch/integration-guides/volcano.md" >}}) をご参照ください。

各 launch エージェントは `max_jobs` パラメータで設定されており、同時に実行できるジョブの最大数を決めます。複数のエージェントは、適切なローンチ用インフラストラクチャーに接続している限り、同じキューを参照できます。

CPU、GPU、メモリなどのリソースの制限は、リソース設定でキューまたはジョブ run レベルで設定可能です。Kubernetes 上でリソース制限付きのキューをセットアップする方法については、[Kubernetes セットアップガイド]({{< relref "/launch/set-up-launch/setup-launch-kubernetes.md" >}}) をご覧ください。

スイープで同時実行される run の数を制限したい場合は、以下のブロックを queue 設定に含めてください。

```yaml title="queue config"
  scheduler:
    num_workers: 4
```