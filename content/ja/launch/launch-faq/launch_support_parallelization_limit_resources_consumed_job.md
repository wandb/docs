---
title: Launch で並列化はサポートされていますか？ジョブが使用するリソースを制限するにはどうすればいいですか？
menu:
  launch:
    identifier: ja-launch-launch-faq-launch_support_parallelization_limit_resources_consumed_job
    parent: launch-faq
---

Launch は複数の GPU やノードにジョブをスケールすることができます。詳細は [Volcano インテグレーションガイド]({{< relref path="/launch/integration-guides/volcano.md" lang="ja" >}}) を参照してください。

各 launch エージェントは `max_jobs` パラメータで設定されており、同時に実行できるジョブの最大数を決定します。複数のエージェントが、適切なローンチ用インフラストラクチャーに接続している限り、単一のキューを共有できます。

CPU、GPU、メモリなどのリソースに関しては、キュー単位やジョブ run 単位で resource 設定内に制限を設けることができます。Kubernetes でリソース制限つきのキューをセットアップする方法については、[Kubernetes セットアップガイド]({{< relref path="/launch/set-up-launch/setup-launch-kubernetes.md" lang="ja" >}}) を参照してください。

スイープで同時実行 run 数を制限する場合は、キュー設定に下記のブロックを追加してください。

```yaml title="queue config"
  scheduler:
    num_workers: 4
```