---
title: W&B にコンテナをビルドしてほしくないのですが、Launch を使うことはできますか？
menu:
  launch:
    identifier: ja-launch-launch-faq-build_container_launch
    parent: launch-faq
---

あらかじめ作成された Docker イメージをローンチするには、以下のコマンドを実行してください。`<>` 内のプレースホルダーは、ご自身の情報に置き換えてください。

```bash
wandb launch -d <docker-image-uri> -q <queue-name> -E <entrypoint>
```

このコマンドはジョブを作成し、run を開始します。

イメージからジョブを作成する場合は、次のコマンドを使用します。

```bash
wandb job create image <image-name> -p <project> -e <entity>
```