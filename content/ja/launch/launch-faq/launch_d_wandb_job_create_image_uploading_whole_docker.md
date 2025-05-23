---
title: '`wandb launch -d` または `wandb job create image` が、レジストリからプルせずに全体のDockerアーティファクトをアップロードしていますか？'
menu:
  launch:
    identifier: ja-launch-launch-faq-launch_d_wandb_job_create_image_uploading_whole_docker
    parent: launch-faq
---

`wandb launch -d` コマンドは、イメージをレジストリにアップロードしません。イメージは別途レジストリにアップロードしてください。以下の手順に従ってください。

1. イメージをビルドします。
2. イメージをレジストリにプッシュします。

ワークフローは以下の通りです：

```bash
docker build -t <repo-url>:<tag> .
docker push <repo-url>:<tag>
wandb launch -d <repo-url>:<tag>
```

ローンチエージェントは、指定されたコンテナを指すジョブを立ち上げます。コンテナレジストリからイメージを取得するエージェントアクセスの設定例については、[Advanced agent setup]({{< relref path="/launch/set-up-launch/setup-agent-advanced.md#agent-configuration" lang="ja" >}})を参照してください。

Kubernetes を使用する場合は、Kubernetes クラスターのポッドが、イメージがプッシュされたレジストリにアクセスできることを確認してください。