---
title: Create alerts and notifications
description: 新しい モデル バージョンが モデルレジストリ にリンクされたときに、Slack 通知を受け取る。
menu:
  default:
    identifier: ja-guides-models-registry-model_registry-notifications
    parent: model-registry
weight: 9
---

新しいモデル バージョンがモデルレジストリにリンクされたときに、Slack 通知を受信します。

1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model) にある W&B Model Registry アプリに移動します。
2. 通知を受信する登録済みモデルを選択します。
3. [**Slack に接続**] ボタンをクリックします。
    {{< img src="/images/models/connect_to_slack.png" alt="" >}}
4. OAuth ページに表示される手順に従って、Slack ワークスペースで W&B を有効にします。

チームの Slack 通知を設定したら、通知を受信する登録済みモデルを選択できます。

{{% alert %}}
チームの Slack 通知が構成されている場合、[**Slack に接続**] ボタンの代わりに [**新しいモデル バージョンが...にリンクされました**] と表示されるトグルが表示されます。
{{% /alert %}}

以下のスクリーンショットは、Slack 通知が設定されている FMNIST classifier 登録済みモデルを示しています。

{{< img src="/images/models/conect_to_slack_fmnist.png" alt="" >}}

新しいモデル バージョンが FMNIST classifier 登録済みモデルにリンクされるたびに、メッセージが接続された Slack チャンネルに自動的に投稿されます。
