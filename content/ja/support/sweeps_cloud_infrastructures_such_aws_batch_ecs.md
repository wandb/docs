---
title: Can you use W&B Sweeps with cloud infrastructures such as AWS Batch, ECS, etc.?
menu:
  support:
    identifier: ja-support-sweeps_cloud_infrastructures_such_aws_batch_ecs
tags:
- sweeps
- aws
toc_hide: true
type: docs
---

`sweep_id` を公開して、W&B Sweep agent がアクセスできるようにするには、これらの エージェント が `sweep_id` を読み取って実行するための メソッド を実装します。

たとえば、Amazon EC2 インスタンスを起動し、その上で `wandb agent` を実行します。SQS キューを使用して、複数の EC2 インスタンスに `sweep_id` をブロードキャストします。各インスタンスは、キューから `sweep_id` を取得して プロセス を開始できます。
