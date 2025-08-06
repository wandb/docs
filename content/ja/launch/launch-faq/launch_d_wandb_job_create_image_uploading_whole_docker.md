---
title: '`wandb launch -d` または `wandb job create image` は、Docker レジストリから取得するのではなく、Docker
  アーティファクト全体をアップロードしているのですか？'
menu:
  launch:
    identifier: launch_d_wandb_job_create_image_uploading_whole_docker
    parent: launch-faq
---

いいえ、`wandb launch -d` コマンドはイメージをレジストリにアップロードしません。イメージはレジストリに別途アップロードしてください。以下の手順に従ってください。

1. イメージをビルドします。
2. イメージをレジストリにプッシュします。

ワークフローは次のようになります。

```bash
docker build -t <repo-url>:<tag> .
docker push <repo-url>:<tag>
wandb launch -d <repo-url>:<tag>
```

ローンチエージェントは、指定されたコンテナを指すジョブを起動します。エージェントがコンテナレジストリからイメージを取得するための設定例については、[Advanced agent setup]({{< relref "/launch/set-up-launch/setup-agent-advanced.md#agent-configuration" >}}) を参照してください。

Kubernetes を利用する場合、Kubernetes クラスターのポッドがイメージをプッシュしたレジストリにアクセスできることを確認してください。