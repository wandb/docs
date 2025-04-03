---
title: I do not want W&B to build a container for me, can I still use Launch?
menu:
  launch:
    identifier: ja-launch-launch-faq-build_container_launch
    parent: launch-faq
---

事前に構築されたDockerイメージをローンチするには、次のコマンドを実行します。`< >`内のプレースホルダーを、お客様固有の情報に置き換えてください。

```bash
wandb launch -d <docker-image-uri> -q <queue-name> -E <entrypoint>
```

このコマンドは、ジョブを作成し、runを開始します。

イメージからジョブを作成するには、次のコマンドを使用します。

```bash
wandb job create image <image-name> -p <project> -e <entity>
```
