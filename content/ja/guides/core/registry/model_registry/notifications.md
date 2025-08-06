---
title: アラートと通知の作成
description: 新しいモデルのバージョンがモデルレジストリにリンクされた際に、Slack 通知を受け取ります。
menu:
  default:
    identifier: notifications
    parent: model-registry
weight: 9
---

新しいモデルバージョンがモデルレジストリにリンクされた際に、Slack 通知を受け取ることができます。

1. [W&B Model Registry app](https://wandb.ai/registry/model) にアクセスします。
2. 通知を受け取りたい Registered Model を選択します。
3. **Connect Slack** ボタンをクリックします。
    {{< img src="/images/models/connect_to_slack.png" alt="Connect to Slack" >}}
4. OAuth ページに表示される手順に従い、W&B を自身の Slack ワークスペースで有効にしてください。

チーム向けに Slack 通知を設定すると、通知を受け取りたい Registered Model を個別に選択できます。

{{% alert %}}
**Connect Slack** ボタンの代わりに **New model version linked to...** というトグルが表示されている場合は、チーム向けにすでに Slack 通知が設定されています。
{{% /alert %}}

下のスクリーンショットは、Slack 通知が有効になっている FMNIST classifier Registered Model の例です。

{{< img src="/images/models/conect_to_slack_fmnist.png" alt="Slack notification example" >}}

FMNIST classifier Registered Model に新しいモデルバージョンがリンクされるたびに、接続された Slack チャンネルに自動的にメッセージが投稿されます。