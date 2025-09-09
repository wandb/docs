---
title: Slack のオートメーションを作成する
menu:
  default:
    identifier: ja-guides-core-automations-create-automations-slack
    parent: create-automations
weight: 1
---

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

このページでは Slack [オートメーション]({{< relref path="/guides/core/automations/" lang="ja" >}}> ) を作成する方法について説明します。webhook オートメーションを作成するには、代わりに [webhook オートメーションを作成する]({{< relref path="/guides/core/automations/create-automations/webhook.md" lang="ja" >}}) を参照してください。

大まかに、Slack オートメーションを作成するには、次の手順を実行します。
1. [Slack インテグレーションを追加する]({{< relref path="#add-a-slack-integration" lang="ja" >}})。これは、W&B が Slack インスタンスおよびチャンネルに投稿することを承認します。
1. [オートメーションを作成する]({{< relref path="#create-an-automation" lang="ja" >}})。これは、監視する [イベント]({{< relref path="/guides/core/automations/automation-events.md" lang="ja" >}}) と通知するチャンネルを定義します。

## Slack インテグレーションを追加する
チーム管理者は、チームに Slack インテグレーションを追加できます。

1. W&B にログインし、**Team Settings (チーム設定)** に移動します。
1. **Slack channel integrations (Slack チャンネルインテグレーション)** セクションで、**Connect Slack (Slack に接続)** をクリックして新しい Slack インスタンスを追加します。既存の Slack インスタンスにチャンネルを追加するには、**New integration (新しいインテグレーション)** をクリックします。

    ![チームに2つの Slack インテグレーションが表示されているスクリーンショット](/images/automations/slack_integrations.png)
1. 必要に応じて、ブラウザで Slack にサインインします。プロンプトが表示されたら、選択した Slack チャンネルに W&B が投稿する権限を付与します。ページを読み、**Search for a channel (チャンネルを検索)** をクリックしてチャンネル名を入力します。リストからチャンネルを選択し、**Allow (許可)** をクリックします。
1. Slack で、選択したチャンネルに移動します。`[ご自身の Slack ハンドル] がこのチャンネルにインテグレーションを追加しました: Weights & Biases` のような投稿が表示された場合、インテグレーションは正しく設定されています。

これで、設定した Slack チャンネルに通知する [オートメーションを作成する]({{< relref path="#create-an-automation" lang="ja" >}}) ことができます。

## Slack インテグレーションを表示および管理する
チーム管理者は、チームの Slack インスタンスとチャンネルを表示および管理できます。

1. W&B にログインし、**Team Settings (チーム設定)** に移動します。
1. **Slack channel integrations (Slack チャンネルインテグレーション)** セクションで、各 Slack の宛先を表示します。
1. 宛先を削除するには、ゴミ箱アイコンをクリックします。

## オートメーションを作成する
[Slack インテグレーションを追加する]({{< relref path="#add-a-slack-integreation" lang="ja" >}}) と、**Registry** または **Project** を選択し、次の手順に従って Slack チャンネルに通知するオートメーションを作成します。

{{< tabpane text=true >}}
{{% tab "Registry" %}}
Registry 管理者は、その Registry でオートメーションを作成できます。

1. W&B にログインします。
1. Registry の名前をクリックして、詳細を表示します。
1. Registry にスコープされたオートメーションを作成するには、**Automations (オートメーション)** タブをクリックし、**Create automation (オートメーションを作成)** をクリックします。Registry にスコープされたオートメーションは、そのすべてのコレクション (将来作成されるものを含む) に自動的に適用されます。

    Registry 内の特定のコレクションにのみスコープされたオートメーションを作成するには、コレクションのアクション `...` メニューをクリックし、**Create automation (オートメーションを作成)** をクリックします。または、コレクションを表示中に、コレクションの詳細ページの **Automations (オートメーション)** セクションにある **Create automation (オートメーションを作成)** ボタンを使用して、そのコレクションのオートメーションを作成します。
1. 監視する [イベント]({{< relref path="/guides/core/automations/automation-events.md" lang="ja" >}}) を選択します。

    表示される追加フィールドに記入します。これらはイベントによって異なります。たとえば、**An artifact alias is added (アーティファクト エイリアスが追加されました)** を選択した場合、**Alias regex (エイリアス正規表現)** を指定する必要があります。

    **Next step (次のステップ)** をクリックします。
1. [Slack インテグレーション]({{< relref path="#add-a-slack-integration" lang="ja" >}}) を所有するチームを選択します。
1. **Action type (アクションの種類)** を **Slack notification (Slack 通知)** に設定します。Slack チャンネルを選択し、**Next step (次のステップ)** をクリックします。
1. オートメーションの名前を入力します。オプションで、説明を入力します。
1. **Create automation (オートメーションを作成)** をクリックします。

{{% /tab %}}
{{% tab "Project" %}}
W&B 管理者は、Project でオートメーションを作成できます。

1. W&B にログインします。
1. Project ページに移動し、**Automations (オートメーション)** タブをクリックし、**Create automation (オートメーションを作成)** をクリックします。

    または、Workspace の線グラフから、表示されているメトリックの [Run メトリック オートメーション]({{< relref path="/guides/core/automations/automation-events.md#run-events" lang="ja" >}}) をすばやく作成できます。パネルにカーソルを合わせ、パネル上部のベルのアイコンをクリックします。
    {{< img src="/images/automations/run_metric_automation_from_panel.png" alt="オートメーションのベルアイコンの位置" >}}
1. 監視する [イベント]({{< relref path="/guides/core/automations/automation-events.md" lang="ja" >}}) を選択します。

    表示される追加フィールドに記入します。これらはイベントによって異なります。たとえば、**An artifact alias is added (アーティファクト エイリアスが追加されました)** を選択した場合、**Alias regex (エイリアス正規表現)** を指定する必要があります。

    **Next step (次のステップ)** をクリックします。
1. [Slack インテグレーション]({{< relref path="#add-a-slack-integration" lang="ja" >}}) を所有するチームを選択します。
1. **Action type (アクションの種類)** を **Slack notification (Slack 通知)** に設定します。Slack チャンネルを選択し、**Next step (次のステップ)** をクリックします。
1. オートメーションの名前を入力します。オプションで、説明を入力します。
1. **Create automation (オートメーションを作成)** をクリックします。

{{% /tab %}}
{{< /tabpane >}}

## オートメーションを表示および管理する

{{< tabpane text=true >}}
{{% tab "Registry" %}}

- Registry のオートメーションは、Registry の **Automations (オートメーション)** タブから管理します。
- コレクションのオートメーションは、コレクションの詳細ページの **Automations (オートメーション)** セクションから管理します。

これらのページのいずれかから、Registry 管理者は既存のオートメーションを管理できます。
- オートメーションの詳細を表示するには、その名前をクリックします。
- オートメーションを編集するには、そのアクション `...` メニューをクリックし、**Edit automation (オートメーションを編集)** をクリックします。
- オートメーションを削除するには、そのアクション `...` メニューをクリックし、**Delete automation (オートメーションを削除)** をクリックします。確認が必要です。


{{% /tab %}}
{{% tab "Project" %}}
W&B 管理者は、Project の **Automations (オートメーション)** タブから Project のオートメーションを表示および管理できます。

- オートメーションの詳細を表示するには、その名前をクリックします。
- オートメーションを編集するには、そのアクション `...` メニューをクリックし、**Edit automation (オートメーションを編集)** をクリックします。
- オートメーションを削除するには、そのアクション `...` メニューをクリックし、**Delete automation (オートメーションを削除)** をクリックします。確認が必要です。
{{% /tab %}}
{{< /tabpane >}}