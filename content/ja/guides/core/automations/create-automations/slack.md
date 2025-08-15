---
title: Slack オートメーションを作成する
menu:
  default:
    identifier: ja-guides-core-automations-create-automations-slack
    parent: create-automations
weight: 1
---

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

このページでは、Slack オートメーションの作成方法について説明します。Webhook オートメーションを作成したい場合は、[Webhook オートメーションの作成]({{< relref path="/guides/core/automations/create-automations/webhook.md" lang="ja" >}}) を参照してください。

大まかな流れとして、Slack オートメーションを作成するには以下の手順を実施します。

1. [Slack インテグレーションを追加]({{< relref path="#add-a-slack-integration" lang="ja" >}})： W&B が指定した Slack インスタンスとチャンネルに投稿できるように認可します。
1. [オートメーションの作成]({{< relref path="#create-an-automation" lang="ja" >}})： 通知対象となる [イベント]({{< relref path="/guides/core/automations/automation-events.md" lang="ja" >}}) とチャンネルを定義します。

## Slack インテグレーションの追加

チーム管理者はチームに Slack インテグレーションを追加できます。

1. W&B にログインし、**Team Settings** に移動します。
1. **Slack channel integrations** セクションで **Connect Slack** をクリックして、新しい Slack インスタンスを追加します。既存の Slack インスタンスにチャンネルを追加したい場合は **New integration** をクリックします。

    ![Screenshot showing two Slack integrations in a Team](/images/automations/slack_integrations.png)
1. 必要に応じてブラウザで Slack にサインインします。プロンプトが表示されたら、W&B に選択した Slack チャンネルへの投稿権限を付与してください。画面をよく読んでから **Search for a channel** をクリックし、チャンネル名を入力し始めてください。リストから対象チャンネルを選択し、**Allow** をクリックします。
1. Slack で選択したチャンネルに移動します。`[あなたの Slack ハンドル] がこのチャンネルにインテグレーションを追加しました: Weights & Biases` のようなメッセージが表示されていれば、インテグレーションは正しく設定されています。

これで、設定した Slack チャンネルへ通知を送る [オートメーションの作成]({{< relref path="#create-an-automation" lang="ja" >}}) ができます。

## Slack インテグレーションの表示・管理

チーム管理者は、そのチームの Slack インスタンスやチャンネルの状況を一覧・管理できます。

1. W&B にログインし、**Team Settings** にアクセスします。
1. **Slack channel integrations** セクションで各 Slack 宛先を確認します。
1. 削除したい宛先のゴミ箱アイコンをクリックすると、その宛先を削除できます。

## オートメーションの作成

[Slack インテグレーションの追加]({{< relref path="#add-a-slack-integration" lang="ja" >}}) 後に、**Registry** または **Project** を選択し、下記手順で Slack チャンネルに通知を行うオートメーションを作成します。

{{< tabpane text=true >}}
{{% tab "Registry" %}}
Registry の管理者は、その Registry 内にオートメーションを作成できます。

1. W&B にログインします。
1. Registry 名をクリックして詳細画面を表示します。
1. Registry 全体に適用されるオートメーションを作成する場合は、**Automations** タブをクリックしてから **Create automation** をクリックします。Registry に紐づくオートメーションは、今後作成されるものも含めてすべてのコレクションに自動的に適用されます。

    Registry 内の特定のコレクションだけにオートメーションを適用したい場合は、対象コレクションのアクション `...` メニューから **Create automation** を選択します。またはコレクションの詳細ページ内 **Automations** セクションの **Create automation** ボタンからも作成できます。
1. 注視する [イベント]({{< relref path="/guides/core/automations/automation-events.md" lang="ja" >}}) を選択します。

    イベントによって表示される追加項目があれば入力します。たとえば、**An artifact alias is added** を選択した場合は **Alias regex** を指定する必要があります。

    **Next step** をクリックします。
1. [Slack インテグレーション]({{< relref path="#add-a-slack-integration" lang="ja" >}}) を所有するチームを選択します。
1. **Action type** を **Slack notification** に設定し、Slack チャンネルを選択後、**Next step** をクリックします。
1. オートメーションの名前を入力し、必要に応じて説明も記入します。
1. **Create automation** をクリックします。

{{% /tab %}}
{{% tab "Project" %}}
W&B の管理者は、プロジェクト内にオートメーションを作成できます。

1. W&B にログインします。
1. プロジェクトページにアクセスし、**Automations** タブをクリックしてから **Create automation** をクリックします。

    あるいは、ワークスペース内のラインプロットから素早く [run metric automation]({{< relref path="/guides/core/automations/automation-events.md#run-events" lang="ja" >}}) を作成することも可能です。パネル上にカーソルを当て、パネル上部のベルアイコンをクリックしてください。
    {{< img src="/images/automations/run_metric_automation_from_panel.png" alt="Automation bell icon location" >}}
1. 注視する [イベント]({{< relref path="/guides/core/automations/automation-events.md" lang="ja" >}}) を選択します。

    イベントによって表示される追加項目があれば入力します。たとえば、**An artifact alias is added** を選択した場合は **Alias regex** を指定する必要があります。

    **Next step** をクリックします。
1. [Slack インテグレーション]({{< relref path="#add-a-slack-integration" lang="ja" >}}) を所有するチームを選択します。
1. **Action type** を **Slack notification** に設定し、Slack チャンネルを選択後、**Next step** をクリックします。
1. オートメーションの名前を入力し、必要に応じて説明も記入します。
1. **Create automation** をクリックします。

{{% /tab %}}
{{< /tabpane >}}

## オートメーションの表示・管理

{{< tabpane text=true >}}
{{% tab "Registry" %}}

- Registry の **Automations** タブから、その Registry のオートメーションを管理できます。
- コレクションの詳細ページ内 **Automations** セクションから、そのコレクションのオートメーションを管理できます。

いずれのページからでも、Registry 管理者は既存オートメーションの管理が可能です。

- オートメーションの詳細を見るには、その名前をクリックします。
- オートメーションを編集するには、アクション `...` メニューから **Edit automation** をクリックします。
- オートメーションを削除するには、アクション `...` メニューから **Delete automation** をクリックします（確認が必要です）。

{{% /tab %}}
{{% tab "Project" %}}
W&B 管理者は、プロジェクトの **Automations** タブからそのプロジェクト内オートメーションを表示・管理できます。

- オートメーションの詳細を見るには、その名前をクリックします。
- オートメーションを編集するには、アクション `...` メニューから **Edit automation** をクリックします。
- オートメーションを削除するには、アクション `...` メニューから **Delete automation** をクリックします（確認が必要です）。
{{% /tab %}}
{{< /tabpane >}}