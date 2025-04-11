---
title: Slack 自動化の作成
menu:
  default:
    identifier: ja-guides-core-automations-create-automations-slack
    parent: create-automations
weight: 1
---

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

このページでは、Slack [オートメーション]({{< relref path="/guides/core/automations/" lang="ja" >}})を作成する方法を示します。ウェブフックオートメーションを作成するには、[ウェブフックオートメーションの作成]({{< relref path="/guides/core/automations/create-automations/webhook.md" lang="ja" >}})を参照してください。

高レベルでは、Slackオートメーションを作成するには、以下の手順を行います:
1. [Slackインテグレーションを追加]({{< relref path="#add-a-slack-channel" lang="ja" >}})し、W&BがSlackインスタンスとチャンネルに投稿できるように認証します。
1. [Slackオートメーションを作成]({{< relref path="#create-slack-automation" lang="ja" >}})し、監視する[イベント]({{< relref path="/guides/core/automations/automation-events.md" lang="ja" >}})と投稿するチャンネルを定義します。

## Slackに接続
チーム管理者は、チームにSlack宛先を追加できます。

1. W&Bにログインし、チーム設定ページに移動します。
1. **Slackチャンネルインテグレーション**セクションで、**Slackを接続**をクリックして新しいSlackインスタンスを追加します。既存のSlackインスタンスにチャンネルを追加するには、**新しいインテグレーション**をクリックします。

    必要に応じて、ブラウザでSlackにサインインします。プロンプトが表示されたら、選択したSlackチャンネルにW&Bからの投稿を許可してください。ページを読み、**チャンネルを検索**をクリックしてチャンネル名を入力し始めます。リストからチャンネルを選択し、**許可**をクリックします。

1. Slackで、選択したチャンネルに移動します。`[あなたのSlackハンドル]がこのチャンネルにインテグレーションを追加しました: Weights & Biases`という投稿が表示されれば、インテグレーションが正しく設定されています。

これで、設定したSlackチャンネルに通知する[オートメーションを作成]({{< relref path="#create-a-slack-automation" lang="ja" >}})できます。

## Slack接続の表示と管理
チーム管理者は、チームのSlackインスタンスとチャンネルを表示および管理できます。

1. W&Bにログインし、**チーム設定**に移動します。
1. **Slackチャンネルインテグレーション**セクションで各Slack宛先を表示します。
1. 宛先を削除するには、そのゴミ箱アイコンをクリックします。

## オートメーションを作成する
W&Bチームを[Slackに接続した後]({{< relref path="#connect-to-slack" lang="ja" >}})、**Registry**または**Project**を選択し、Slackチャンネルに通知するオートメーションを作成する手順に従います。

{{< tabpane text=true >}}
{{% tab "Registry" %}}
Registry管理者は、そのregistry内でオートメーションを作成できます。

1. W&Bにログインします。
1. registryの名前をクリックして詳細を表示します。
1. registryに関連付けられたオートメーションを作成するには、**Automations**タブをクリックし、**Create automation**をクリックします。registryに関連付けられたオートメーションは、自動的にそのすべてのコレクションに適用されます（将来作成されるものも含む）。

    registry内の特定のコレクションにのみ関連付けられたオートメーションを作成するには、コレクションのアクション`...`メニューをクリックし、**Create automation**をクリックします。あるいは、コレクションを閲覧しているときに、**Automations**セクションの詳細ページで**Create automation**ボタンを使用して、そのためのオートメーションを作成します。
1. 監視する[**Event**]({{< relref path="/guides/core/automations/automation-events.md" lang="ja" >}})を選択します。

    イベントに応じた追加のフィールドが表示されるので、それを入力します。たとえば、**An artifact alias is added**を選択した場合は、**Alias regex**を指定する必要があります。
    
    **Next step**をクリックします。
1. [Slackインテグレーション]({{< relref path="#add-a-slack-integration" lang="ja" >}})を所有するチームを選択します。
1. **Action type**を**Slack notification**に設定します。Slackチャンネルを選択し、**Next step**をクリックします。
1. オートメーションの名前を提供します。必要に応じて、説明を追加します。
1. **Create automation**をクリックします。

{{% /tab %}}
{{% tab "Project" %}}
W&B管理者は、プロジェクト内でオートメーションを作成できます。

1. W&Bにログインします。
1. プロジェクトページに移動し、**Automations**タブをクリックします。
1. **Create automation**をクリックします。
1. 監視する[**Event**]({{< relref path="/guides/core/automations/automation-events.md" lang="ja" >}})を選択します。

    イベントに応じた追加のフィールドが表示されるので、それを入力します。たとえば、**An artifact alias is added**を選択した場合は、**Alias regex**を指定する必要があります。
    
    **Next step**をクリックします。
1. [Slackインテグレーション]({{< relref path="#add-a-slack-integration" lang="ja" >}})を所有するチームを選択します。
1. **Action type**を**Slack notification**に設定します。Slackチャンネルを選択し、**Next step**をクリックします。
1. オートメーションの名前を提供します。必要に応じて、説明を追加します。
1. **Create automation**をクリックします。

{{% /tab %}}
{{< /tabpane >}}

## オートメーションの表示と管理

{{< tabpane text=true >}}
{{% tab "Registry" %}}

- registryの**Automations**タブから対象のオートメーションを管理します。
- コレクションの詳細ページの**Automations**セクションからコレクションのオートメーションを管理します。

これらのページのいずれからも、Registry管理者は既存のオートメーションを管理できます:
- オートメーションの詳細を表示するには、その名前をクリックします。
- オートメーションを編集するには、そのアクション`...`メニューをクリックし、**Edit automation**をクリックします。
- オートメーションを削除するには、そのアクション`...`メニューをクリックし、**Delete automation**をクリックします。確認が必要です。

{{% /tab %}}
{{% tab "Project" %}}
W&B管理者は、プロジェクトの**Automations**タブからプロジェクトのオートメーションを表示および管理できます。

- オートメーションの詳細を表示するには、その名前をクリックします。
- オートメーションを編集するには、そのアクション`...`メニューをクリックし、**Edit automation**をクリックします。
- オートメーションを削除するには、そのアクション`...`メニューをクリックし、**Delete automation**をクリックします。確認が必要です。
{{% /tab %}}
{{< /tabpane >}}