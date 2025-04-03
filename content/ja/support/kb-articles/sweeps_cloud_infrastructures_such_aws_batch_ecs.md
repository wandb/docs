---
title: Can you use W&B Sweeps with cloud infrastructures such as AWS Batch, ECS, etc.?
menu:
  support:
    identifier: ja-support-kb-articles-sweeps_cloud_infrastructures_such_aws_batch_ecs
support:
- sweeps
- aws
toc_hide: true
type: docs
url: /support/:filename
---

`sweep_id` を公開して、どの W&B Sweep agent でもアクセスできるようにするには、これらの agent が `sweep_id` を読み取って実行するための method を実装します。

たとえば、Amazon EC2 インスタンスをローンンチし、そこで `wandb agent` を実行します。SQS キューを使用して、`sweep_id` を複数の EC2 インスタンスにブロードキャストします。各インスタンスは、キューから `sweep_id` を取得して、プロセスを開始できます。