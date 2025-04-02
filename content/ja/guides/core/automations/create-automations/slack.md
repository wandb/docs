---
title: Create a Slack automation
menu:
  default:
    identifier: ja-guides-core-automations-create-automations-slack
    parent: create-automations
weight: 1
---

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

このページでは、Slack [ オートメーション ]({{< relref path="/guides/core/automations/" lang="ja" >}}) の作成方法について説明します。Webhook オートメーションを作成するには、代わりに [Webhook オートメーションの作成]({{< relref path="/guides/core/automations/create-automations/webhook.md" lang="ja" >}}) を参照してください。

大まかに言うと、Slack オートメーションを作成するには、次の手順を実行します。

1. [Slack インテグレーションの追加]({{< relref path="#add-a-slack-channel" lang="ja" >}}) 。これにより、Weights & Biases が Slack インスタンスと channel に投稿することを承認します。
2. [Slack オートメーションの作成]({{< relref path="#create-slack-automation" lang="ja" >}}) 。これにより、監視する [event]({{< relref path="/guides/core/automations/automation-events.md" lang="ja" >}}) と投稿先の channel が定義されます。

## Slack に接続する
Team の管理者は、Team に Slack の送信先を追加できます。

1. Weights & Biases にログインして、Team Settings ページに移動します。
2. [**Slack channel integrations**] セクションで、[**Connect Slack**] をクリックして新しい Slack インスタンスを追加します。既存の Slack インスタンスに channel を追加するには、[**New integration**] をクリックします。

    必要に応じて、ブラウザで Slack にサインインします。プロンプトが表示されたら、選択した Slack channel に投稿する権限を Weights & Biases に付与します。ページを読んでから、[**Search for a channel**] をクリックして、channel 名を入力します。リストから channel を選択し、[**Allow**] をクリックします。

3. Slack で、選択した channel に移動します。`[Your Slack handle] added an integration to this channel: Weights & Biases` のような投稿が表示された場合、インテグレーションは正しく構成されています。

これで、構成した Slack channel に通知する [ オートメーションの作成 ]({{< relref path="#create-a-slack-automation" lang="ja" >}}) ができます。

## Slack 接続の表示と管理
Team の管理者は、Team の Slack インスタンスと channel を表示および管理できます。

1. Weights & Biases にログインして、[**Team Settings**] に移動します。
2. [**Slack channel integrations**] セクションで、各 Slack の送信先を表示します。
3. 送信先を削除するには、ゴミ箱アイコンをクリックします。

## オートメーションの作成
[Weights & Biases Team を Slack に接続]({{< relref path="#connect-to-slack" lang="ja" >}}) したら、[**Registry**] または [**Project**] を選択し、次の手順に従って Slack channel に通知するオートメーションを作成します。

{{< tabpane text=true >}}
{{% tab "Registry" %}}
Registry の管理者は、その Registry でオートメーションを作成できます。

1. Weights & Biases にログインします。
2. Registry の名前をクリックして、詳細を表示します。
3. Registry を対象範囲とするオートメーションを作成するには、[**Automations**] タブをクリックし、[**Create automation**] をクリックします。Registry を対象範囲とするオートメーションは、そのすべてのコレクション (将来作成されるコレクションを含む) に自動的に適用されます。

    Registry 内の特定のコレクションのみを対象範囲とするオートメーションを作成するには、コレクションのアクション `...` メニューをクリックし、[**Create automation**] をクリックします。または、コレクションを表示しているときに、コレクションの詳細ページの [**Automations**] セクションにある [**Create automation**] ボタンを使用して、コレクションのオートメーションを作成します。
4. 監視する [**Event**]({{< relref path="/guides/core/automations/automation-events.md" lang="ja" >}}) を選択します。

    表示される追加フィールドに入力します。これは、イベントによって異なります。たとえば、[**An artifact alias is added**] を選択した場合は、[**Alias regex**] を指定する必要があります。

    [**Next step**] をクリックします。
5. [Slack インテグレーション]({{< relref path="#add-a-slack-integration" lang="ja" >}}) を所有する Team を選択します。
6. [**Action type**] を [**Slack notification**] に設定します。Slack channel を選択し、[**Next step**] をクリックします。
7. オートメーションの名前を入力します。必要に応じて、説明を入力します。
8. [**Create automation**] をクリックします。

{{% /tab %}}
{{% tab "Project" %}}
Weights & Biases の管理者は、Project でオートメーションを作成できます。

1. Weights & Biases にログインします。
2. Project ページに移動し、[**Automations**] タブをクリックします。
3. [**Create automation**] をクリックします。
4. 監視する [**Event**]({{< relref path="/guides/core/automations/automation-events.md" lang="ja" >}}) を選択します。

    表示される追加フィールドに入力します。これは、イベントによって異なります。たとえば、[**An artifact alias is added**] を選択した場合は、[**Alias regex**] を指定する必要があります。

    [**Next step**] をクリックします。
5. [Slack インテグレーション]({{< relref path="#add-a-slack-integration" lang="ja" >}}) を所有する Team を選択します。
6. [**Action type**] を [**Slack notification**] に設定します。Slack channel を選択し、[**Next step**] をクリックします。
7. オートメーションの名前を入力します。必要に応じて、説明を入力します。
8. [**Create automation**] をクリックします。

{{% /tab %}}
{{< /tabpane >}}

## オートメーションの表示と管理

{{< tabpane text=true >}}
{{% tab "Registry" %}}

- Registry のオートメーションは、Registry の [**Automations**] タブから管理します。
- コレクションのオートメーションは、コレクションの詳細ページの [**Automations**] セクションから管理します。

これらのページのいずれかから、Registry の管理者は既存のオートメーションを管理できます。

- オートメーションの詳細を表示するには、その名前をクリックします。
- オートメーションを編集するには、アクション `...` メニューをクリックし、[**Edit automation**] をクリックします。
- オートメーションを削除するには、アクション `...` メニューをクリックし、[**Delete automation**] をクリックします。確認が必要です。

{{% /tab %}}
{{% tab "Project" %}}
Weights & Biases の管理者は、Project の [**Automations**] タブから Project のオートメーションを表示および管理できます。

- オートメーションの詳細を表示するには、その名前をクリックします。
- オートメーションを編集するには、アクション `...` メニューをクリックし、[**Edit automation**] をクリックします。
- オートメーションを削除するには、アクション `...` メニューをクリックし、[**Delete automation**] をクリックします。確認が必要です。
{{% /tab %}}
{{< /tabpane >}}
