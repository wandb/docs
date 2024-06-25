---
description: 新しいモデルバージョンがモデルレジストリにリンクされたときにSlack通知を受け取る。
displayed_sidebar: default
---


# アラートと通知の作成

新しいモデルバージョンがモデルレジストリにリンクされたときに、Slack通知を受け取ります。

1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model) で W&B Model Registry アプリに移動します。
2. 通知を受け取りたい登録済みのモデルを選択します。
3. **Connect Slack** ボタンをクリックします。
    ![](/images/models/connect_to_slack.png)
4. OAuth ページに表示される指示に従って、Slack ワークスペースで W&B を有効にします。

チームのために Slack 通知を設定したら、通知を受け取りたい登録済みモデルを選択できます。

:::info
Slack 通知がチームのために設定されている場合、**Connect Slack** ボタンの代わりに **New model version linked to...** というメッセージが表示されます。
:::

以下のスクリーンショットは、Slack 通知が設定されている FMNIST 分類器の登録済みモデルを示しています。

![](/images/models/conect_to_slack_fmnist.png)

新しいモデルバージョンが FMNIST 分類器の登録済みモデルにリンクされるたびに、接続された Slack チャンネルにメッセージが自動的に投稿されます。