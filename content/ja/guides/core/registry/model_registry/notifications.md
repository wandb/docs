---
title: アラートと通知を作成する
description: 新しい Model バージョンが Model Registry にリンクされたときに Slack 通知を受け取ります。
menu:
  default:
    identifier: ja-guides-core-registry-model_registry-notifications
    parent: model-registry
weight: 9
---

新しいモデルバージョンがモデルレジストリにリンクされたときに、Slack 通知を受け取れるように設定します。

1. [W&B モデルレジストリ アプリ](https://wandb.ai/registry/model)に移動します。
2. 通知を受け取りたい Registered Model を選択します。
3. **Connect Slack** ボタンをクリックします。
    {{< img src="/images/models/connect_to_slack.png" alt="Slack に接続" >}}
4. OAuth ページに表示される手順に従い、Slack ワークスペースで W&B を有効にします。

Teams の Slack 通知を設定すると、通知を受け取りたい Registered Model を自由に選択できます。

{{% alert %}}
Teams の Slack 通知が設定されている場合、**Connect Slack** ボタンの代わりに、**New model version linked to...** と表示されたトグルが表示されます。
{{% /alert %}}

以下のスクリーンショットは、Slack 通知が設定された FMNIST 分類器の Registered Model を示しています。

{{< img src="/images/models/conect_to_slack_fmnist.png" alt="Slack 通知の例" >}}

新しいモデルバージョンが FMNIST 分類器の Registered Model にリンクされるたびに、接続された Slack チャンネルにメッセージが自動的に投稿されます。