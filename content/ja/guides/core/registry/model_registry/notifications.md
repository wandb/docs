---
title: アラートと通知の作成
description: 新しいモデルのバージョンがモデルレジストリにリンクされたときに、Slack 通知を受け取ります。
menu:
  default:
    identifier: ja-guides-core-registry-model_registry-notifications
    parent: model-registry
weight: 9
---

新しいモデルバージョンがモデルレジストリにリンクされたとき、Slack 通知を受け取ることができます。

1. [W&B Model Registry アプリ](https://wandb.ai/registry/model) にアクセスします。
2. 通知を受け取りたい Registered Model を選択します。
3. **Connect Slack** ボタンをクリックします。
    {{< img src="/images/models/connect_to_slack.png" alt="Connect to Slack" >}}
4. OAuth ページに表示される手順に従い、Slack の Workspace で W&B を有効化してください。

Slack 通知をチーム向けに設定した後は、通知を受けたい Registered Model を個別に選ぶことができます。

{{% alert %}}
チームで Slack 通知が設定されている場合、**Connect Slack** ボタンの代わりに **New model version linked to...** というトグルが表示されます。
{{% /alert %}}

下記のスクリーンショットは、Slack 通知が有効になっている FMNIST classifier の Registered Model の例です。

{{< img src="/images/models/conect_to_slack_fmnist.png" alt="Slack notification example" >}}

新しいモデルバージョンが FMNIST classifier の Registered Model にリンクされるたび、自動的に接続された Slack チャンネルにメッセージが投稿されます。