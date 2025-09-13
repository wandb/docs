---
title: '`wandb launch -d` や `wandb job create image` は、レジストリから pull するのではなく、Docker
  アーティファクトを丸ごとアップロードしているのですか？'
menu:
  launch:
    identifier: ja-launch-launch-faq-launch_d_wandb_job_create_image_uploading_whole_docker
    parent: launch-faq
---

いいえ、 `wandb launch -d` コマンドはイメージをレジストリにアップロードしません。イメージはレジストリに別途アップロードしてください。次の手順に従ってください:

1. イメージをビルドする。
2. イメージをレジストリにプッシュする。

ワークフローは次のとおりです:

```bash
docker build -t <repo-url>:<tag> .
docker push <repo-url>:<tag>
wandb launch -d <repo-url>:<tag>
```

その後、Launch エージェントが指定したコンテナを参照するジョブを起動します。コンテナ レジストリからイメージをプルするためのエージェントのアクセス設定例については、[エージェントの高度なセットアップ]({{< relref path="/launch/set-up-launch/setup-agent-advanced.md#agent-configuration" lang="ja" >}}) を参照してください。

Kubernetes の場合、イメージをプッシュしたレジストリに Kubernetes クラスターの Pod がアクセスできることを確認してください。