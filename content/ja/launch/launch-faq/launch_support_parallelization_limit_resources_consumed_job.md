---
title: Launch は並列実行をサポートしていますか？ジョブが消費するリソースを制限するにはどうすればよいですか？
menu:
  launch:
    identifier: ja-launch-launch-faq-launch_support_parallelization_limit_resources_consumed_job
    parent: launch-faq
---

Launch は複数の GPU とノードにまたがるジョブのスケーリングをサポートします。詳細は [Volcano インテグレーション ガイド]({{< relref path="/launch/integration-guides/volcano.md" lang="ja" >}}) を参照してください。

各 Launch エージェントには `max_jobs` パラメータが設定されており、同時に実行できるジョブの最大数が決まります。適切な Launch のインフラストラクチャーに接続していれば、複数のエージェントを 1 つのキューに向けることができます。

リソース設定で、キュー レベルまたはジョブの run レベルで CPU、GPU、メモリなどの制限を設定できます。Kubernetes でリソース制限付きのキューをセットアップする方法については、[Kubernetes セットアップ ガイド]({{< relref path="/launch/set-up-launch/setup-launch-kubernetes.md" lang="ja" >}}) を参照してください。

Sweeps の場合、同時実行される run 数を制限するため、キュー設定に次のブロックを含めてください。

```yaml title="キュー設定"
  scheduler:
    num_workers: 4
```