---
title: 私が W&B にコンテナを作成してほしくない場合でも、Launch を使用できますか？
menu:
  launch:
    identifier: ja-launch-launch-faq-build_container_launch
    parent: launch-faq
---

事前に構築された Docker イメージを起動するには、以下のコマンドを実行してください。`<>` 内のプレースホルダーを具体的な情報に置き換えてください：

```bash
wandb launch -d <docker-image-uri> -q <queue-name> -E <entrypoint>
```

このコマンドはジョブを作成し、run を開始します。

イメージからジョブを作成するには、以下のコマンドを使用してください：

```bash
wandb job create image <image-name> -p <project> -e <entity>
```