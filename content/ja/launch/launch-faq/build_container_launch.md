---
title: I do not want W&B to build a container for me, can I still use Launch?
menu:
  launch:
    identifier: ja-launch-launch-faq-build_container_launch
    parent: launch-faq
---

事前に構築された Docker イメージを ローンチ するには、次の コマンド を実行します。`<>` 内のプレースホルダーを、特定の情報に置き換えてください。

```bash
wandb launch -d <docker-image-uri> -q <queue-name> -E <entrypoint>
```

この コマンド は、ジョブを作成し、run を開始します。

イメージからジョブを作成するには、次の コマンド を使用します。

```bash
wandb job create image <image-name> -p <project> -e <entity>
```
