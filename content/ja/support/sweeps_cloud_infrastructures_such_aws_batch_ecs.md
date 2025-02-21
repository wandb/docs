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

`sweep_id` を公開し、すべての W&B sweep agent が アクセス できるようにするには、これらのエージェントが `sweep_id` を読み取り、実行する メソッド を実装します。

たとえば、Amazon EC2 インスタンスをロールーチして、そこで `wandb agent` を実行します。SQS キューを使用して、`sweep_id` を複数の EC2 インスタンスにブロードキャストします。それぞれのインスタンスはキューから `sweep_id` を取得し、プロセスを開始できます。