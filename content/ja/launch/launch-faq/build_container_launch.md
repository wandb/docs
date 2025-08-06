---
title: 自分でコンテナを作りたい場合でも Launch を使えますか？
menu:
  launch:
    identifier: build_container_launch
    parent: launch-faq
---

事前に用意された Docker イメージをローンチするには、以下のコマンドを実行してください。`<>` のプレースホルダーをあなたの情報に置き換えてください。

```bash
wandb launch -d <docker-image-uri> -q <queue-name> -E <entrypoint>
```

このコマンドはジョブを作成し、run を開始します。

イメージからジョブを作成するには、次のコマンドを使用します。

```bash
wandb job create image <image-name> -p <project> -e <entity>
```