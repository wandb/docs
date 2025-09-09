---
title: AWS Batch や ECS などの クラウド インフラストラクチャー で W&B Sweeps を使用できますか？
menu:
  support:
    identifier: ja-support-kb-articles-sweeps_cloud_infrastructures_such_aws_batch_ecs
support:
- Sweeps
- AWS
toc_hide: true
type: docs
url: /support/:filename
---

任意の W&B Sweep agent が アクセス できるように `sweep_id` を公開するには、これらの エージェント が `sweep_id` を読み取り、実行できる メソッド を実装します。

例えば、Amazon EC2 インスタンスを起動して、その上で `wandb agent` を実行します。SQS キューを使って、`sweep_id` を複数の EC2 インスタンスに配信します。各インスタンスはキューから `sweep_id` を取得し、プロセスを開始できます。