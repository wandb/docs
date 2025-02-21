---
title: Is `wandb launch -d` or `wandb job create image` uploading a whole docker artifact
  and not pulling from a registry?
menu:
  launch:
    identifier: ja-launch-launch-faq-launch_d_wandb_job_create_image_uploading_whole_docker
    parent: launch-faq
---

No, `wandb launch -d` コマンドはイメージをレジストリにアップロードしません。イメージは別途レジストリにアップロードしてください。以下の手順に従ってください。

1. イメージをビルドします。
2. イメージをレジストリにプッシュします。

ワークフローは次のようになります:

```bash
docker build -t <repo-url>:<tag> .
docker push <repo-url>:<tag>
wandb launch -d <repo-url>:<tag>
```

ローンチ エージェントは、指定されたコンテナを指すジョブをスピンアップします。コンテナ レジストリからイメージをプルするためのエージェント アクセスを設定する例については、[高度なエージェント設定]({{< relref path="/launch/set-up-launch/setup-agent-advanced.md#agent-configuration" lang="ja" >}}) を参照してください。

Kubernetes の場合、Kubernetes クラスターのポッドがイメージをプッシュしたレジストリにアクセスできることを確認してください。