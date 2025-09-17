---
title: W&B にコンテナをビルドしてほしくありません。Launch はそれでも使えますか？
menu:
  launch:
    identifier: ja-launch-launch-faq-build_container_launch
    parent: launch-faq
---

事前構築済みの Docker イメージを起動するには、次のコマンドを実行します。`<>` のプレースホルダーをご自身の情報に置き換えてください:

```bash
wandb launch -d <docker-image-uri> -q <queue-name> -E <entrypoint>
```

このコマンドはジョブを作成し、run を開始します。

イメージからジョブを作成するには、次のコマンドを使用します:

```bash
wandb job create image <image-name> -p <project> -e <entity>
```