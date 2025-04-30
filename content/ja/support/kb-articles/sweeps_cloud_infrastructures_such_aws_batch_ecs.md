---
title: クラウド インフラストラクチャー（AWS Batch、ECS など）で W&B スイープを使用できますか？
menu:
  support:
    identifier: ja-support-kb-articles-sweeps_cloud_infrastructures_such_aws_batch_ecs
support:
- sweeps
- aws
toc_hide: true
type: docs
url: /ja/support/:filename
---

`W&B` sweep エージェントがアクセスできるようにするために、これらのエージェントが `sweep_id` を読み取り、実行するためのメソッドを実装します。

例えば、Amazon EC2 インスタンスをローンチし、その上で `wandb agent` を実行します。SQS キューを使用して `sweep_id` を複数の EC2 インスタンスにブロードキャストします。各インスタンスはその後、キューから `sweep_id` を取得し、プロセスを開始することができます。