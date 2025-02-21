---
title: I do not want W&B to build a container for me, can I still use Launch?
menu:
  launch:
    identifier: ja-launch-launch-faq-build_container_launch
    parent: launch-faq
---

プレビルドの Docker イメージを使用して ローンンチ するには、次の コマンド を実行してください。 `< >` のプレースホルダをあなたの特定の情報で置き換えてください。

```bash
wandb launch -d <docker-image-uri> -q <queue-name> -E <entrypoint>
```

この コマンド はジョブを作成し、run を開始します。

イメージからジョブを作成するには、次の コマンド を使用してください。

```bash
wandb job create image <image-name> -p <project> -e <entity>
```