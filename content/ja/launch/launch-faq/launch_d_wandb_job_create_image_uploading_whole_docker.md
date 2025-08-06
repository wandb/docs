---
title: '`wandb launch -d` や `wandb job create image` は、Docker レジストリから取得するのではなく、Docker
  全体をアーティファクトとしてアップロードしているのでしょうか？'
menu:
  launch:
    identifier: ja-launch-launch-faq-launch_d_wandb_job_create_image_uploading_whole_docker
    parent: launch-faq
---

いいえ、`wandb launch -d` コマンドはイメージをレジストリにアップロードしません。イメージは別途レジストリにアップロードしてください。手順は以下の通りです。

1. イメージをビルドする。
2. イメージをレジストリにプッシュする。

ワークフローは以下のようになります。

```bash
docker build -t <repo-url>:<tag> .
docker push <repo-url>:<tag>
wandb launch -d <repo-url>:<tag>
```

ローンチ エージェントは、指定したコンテナを指すジョブを起動します。コンテナ レジストリからイメージをプルするためのエージェントのアクセス設定例については、[Advanced agent setup]({{< relref path="/launch/set-up-launch/setup-agent-advanced.md#agent-configuration" lang="ja" >}}) を参照してください。

Kubernetes を使用する場合は、Kubernetes クラスターのポッドが、イメージをプッシュしたレジストリにアクセスできることを確認してください。