---
title: Create alerts and notifications
description: 新しい モデル バージョン が モデルレジストリ にリンクされたときに Slack 通知を受け取ります。
menu:
  default:
    identifier: ja-guides-core-registry-model_registry-notifications
    parent: model-registry
weight: 9
---

新しいモデル バージョンがモデルレジストリにリンクされたときに、Slack通知を受信します。

1. W&B モデルレジストリのアプリ（[https://wandb.ai/registry/model](https://wandb.ai/registry/model)）に移動します。
2. 通知を受信したい Registered Model を選択します。
3. [**Connect Slack**] ボタンをクリックします。
    {{< img src="/images/models/connect_to_slack.png" alt="" >}}
4. OAuth ページに表示される手順に従って、Slack workspace で W&B を有効にします。

チームの Slack 通知を設定すると、通知を受信する Registered Model を選択できます。

{{% alert %}}
チームの Slack 通知が設定されている場合、[**Connect Slack**] ボタンの代わりに、[**New model version linked to...**] と表示されるトグルが表示されます。
{{% /alert %}}

以下のスクリーンショットは、Slack 通知が設定されている FMNIST 分類器の Registered Model を示しています。

{{< img src="/images/models/conect_to_slack_fmnist.png" alt="" >}}

新しいモデル バージョンが FMNIST 分類器の Registered Model にリンクされるたびに、メッセージが接続された Slack チャンネルに自動的に投稿されます。