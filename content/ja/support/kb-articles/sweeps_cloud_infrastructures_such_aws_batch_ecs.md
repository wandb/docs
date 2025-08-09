---
title: W&B Sweeps は AWS Batch や ECS などのクラウドインフラストラクチャーと一緒に使えますか？
menu:
  support:
    identifier: ja-support-kb-articles-sweeps_cloud_infrastructures_such_aws_batch_ecs
support:
- スイープ
- 'aws

  '
toc_hide: true
type: docs
url: /support/:filename
---

`sweep_id` を公開して、どの W&B Sweep agent からもアクセスできるようにするには、これらのエージェントが `sweep_id` を読み取り実行できるメソッドを実装してください。

例えば、Amazon EC2 インスタンスをローンチし、その上で `wandb agent` を実行します。SQS キューを使って `sweep_id` を複数の EC2 インスタンスにブロードキャストします。各インスタンスはキューから `sweep_id` を取得し、プロセスを開始できます。