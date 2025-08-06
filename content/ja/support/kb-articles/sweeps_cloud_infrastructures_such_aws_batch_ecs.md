---
title: W&B Sweeps は AWS Batch や ECS などのクラウドインフラストラクチャーと一緒に利用できますか？
url: /support/:filename
toc_hide: true
type: docs
support:
- スイープ
- aws
---

`sweep_id` を公開して、どの W&B Sweep agent でも アクセス できるようにするためには、これらのエージェントが `sweep_id` を読み取り、実行できるメソッドを実装してください。

例えば、Amazon EC2 インスタンスをローンンチし、その上で `wandb agent` を実行します。SQS キューを使用して複数の EC2 インスタンスに `sweep_id` を配信します。各インスタンスはキューから `sweep_id` を取得し、プロセスを開始できます。