---
title: Slack オートメーションを作成する
menu:
  default:
    identifier: create-slack-automations
    parent: create-automations
weight: 1
---

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

このページでは、Slack オートメーションの作成方法を説明します。Webhook オートメーションを作成したい場合は、[Webhook オートメーションの作成]({{< relref "/guides/core/automations/create-automations/webhook.md" >}}) をご参照ください。

Slack オートメーションを作成する流れは次の通りです。

1. [Slack インテグレーションを追加]({{< relref "#add-a-slack-integration" >}})します。これにより W&B が指定した Slack インスタンスやチャンネルへの投稿権限を持てます。
2. [オートメーションを作成]({{< relref "#create-an-automation" >}})します。これには監視したい [event]({{< relref "/guides/core/automations/automation-events.md" >}}) と通知するチャンネルの指定を含みます。

## Slack インテグレーションを追加する
Team の管理者が Slack インテグレーションを追加できます。

1. W&B にログインし、**Team Settings** に移動します。
2. **Slack channel integrations** セクションで **Connect Slack** をクリックし、新しく Slack インスタンスを追加します。すでに追加済みの Slack インスタンスに新しいチャンネルを追加したい場合は **New integration** をクリックしてください。

    ![Screenshot showing two Slack integrations in a Team](/images/automations/slack_integrations.png)
3. 必要に応じて、ブラウザで Slack にサインインします。プロンプトが表示されたら、W&B に選択した Slack チャンネルへの投稿権限を付与してください。画面の案内を読み、**Search for a channel** をクリックし、チャンネル名の入力を始めます。リストからチャンネルを選び、**Allow** をクリックします。
4. Slack で選択したチャンネルに移動します。`[あなたの Slack ハンドル] added an integration to this channel: Weights & Biases` という投稿が見えれば、インテグレーションは正しく設定されています。

これで、設定した Slack チャンネルに通知する [オートメーションを作成]({{< relref "#create-an-automation" >}}) できます。

## Slack インテグレーションの確認と管理
Team 管理者は Team の Slack インスタンスやチャンネルを確認・管理できます。

1. W&B にログインし、**Team Settings** に移動します。
2. **Slack channel integrations** セクションで、各 Slack 宛先を確認します。
3. 削除したい場合は、ごみ箱アイコンをクリックして宛先を削除します。

## オートメーションを作成する
[Slack インテグレーションを追加]({{< relref "#add-a-slack-integreation" >}}) 後に、**Registry** または **Project** を選び、次の手順に沿って Slack チャンネルに通知するオートメーションを作成します。

{{< tabpane text=true >}}
{{% tab "Registry" %}}
Registry 管理者は、その Registry にオートメーションを作成できます。

1. W&B にログインします。
2. Registry 名をクリックし、詳細ページを表示します。
3. Registry 単位でオートメーションを作成するには、**Automations** タブをクリックし、**Create automation** をクリックします。Registry 単位のオートメーションは、その Registry 配下のすべてのコレクション（将来作成されるものも含む）に自動的に適用されます。

    Registry 内の特定のコレクションのみに限定したオートメーションを作成する場合は、コレクションのアクション `...` メニューから **Create automation** を選択します。または、コレクション詳細ページの **Automations** セクションで **Create automation** ボタンから作成できます。
4. 監視する [event]({{< relref "/guides/core/automations/automation-events.md" >}}) を選びます。

    event に応じて追加の入力フィールドが表示されます。例えば **An artifact alias is added** を選んだ場合は **Alias regex** の指定が必要です。

    **Next step** をクリックします。
5. [Slack インテグレーション]({{< relref "#add-a-slack-integration" >}}) を所有する Team を選択します。
6. **Action type** を **Slack notification** に設定します。Slack チャンネルを選択し、**Next step** をクリックします。
7. オートメーション名を入力します。必要に応じて説明も記入できます。
8. **Create automation** をクリックします。

{{% /tab %}}
{{% tab "Project" %}}
W&B 管理者は Project 内にもオートメーションを作成できます。

1. W&B にログインします。
2. プロジェクトページに移動し、**Automations** タブをクリックしてから **Create automation** をクリックします。

   または、ワークスペース内の折れ線グラフパネルから、表示されている指標に対してすぐに [run metric automation]({{< relref "/guides/core/automations/automation-events.md#run-events" >}}) を作成できます。パネルにカーソルを合わせ、パネル上部のベルアイコンをクリックします。
   {{< img src="/images/automations/run_metric_automation_from_panel.png" alt="Automation bell icon location" >}}
3. 監視する [event]({{< relref "/guides/core/automations/automation-events.md" >}}) を選びます。

    event に応じて追加の入力フィールドが表示されます。例えば **An artifact alias is added** を選んだ場合は **Alias regex** の指定が必要です。

    **Next step** をクリックします。
4. [Slack インテグレーション]({{< relref "#add-a-slack-integration" >}}) を所有する Team を選択します。
5. **Action type** を **Slack notification** に設定します。Slack チャンネルを選択し、**Next step** をクリックします。
6. オートメーション名を入力します。必要に応じて説明も記入できます。
7. **Create automation** をクリックします。

{{% /tab %}}
{{< /tabpane >}}

## オートメーションの閲覧・管理

{{< tabpane text=true >}}
{{% tab "Registry" %}}

- Registry の **Automations** タブから、その Registry のオートメーションを管理します。
- コレクション単体の場合は、コレクション詳細ページの **Automations** セクションから管理します。

これらのページから Registry 管理者は既存のオートメーションを管理できます：
- オートメーションの詳細を確認するには、名前をクリックします。
- 編集するには、アクション `...` メニューから **Edit automation** を選択します。
- 削除するには、アクション `...` メニューから **Delete automation** を選択します（確認が求められます）。

{{% /tab %}}
{{% tab "Project" %}}
W&B 管理者は、そのプロジェクトの **Automations** タブからプロジェクトのオートメーションを閲覧・管理できます。

- オートメーションの詳細を確認するには、名前をクリックします。
- 編集するには、アクション `...` メニューから **Edit automation** を選択します。
- 削除するには、アクション `...` メニューから **Delete automation** を選択します（確認が求められます）。
{{% /tab %}}
{{< /tabpane >}}