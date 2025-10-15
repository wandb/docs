---
title: Webhook オートメーションを作成する
menu:
  default:
    identifier: ja-guides-core-automations-create-automations-webhook
    parent: automations
weight: 3
---

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

このページでは、webhook のオートメーションを作成する方法を示します。Slack オートメーションを作成するには、代わりに [Slack オートメーションの作成]({{< relref path="/guides/core/automations/create-automations/slack.md" lang="ja" >}})を参照してください。

webhook オートメーションを作成するための大まかな手順は以下の通りです。

1. 必要に応じて、オートメーションに必要なアクストークン、パスワード、またはSSHキーなどを含む機密文字列ごとに[W&B シークレットを作成]({{< relref path="/guides/core/secrets.md" lang="ja" >}})します。シークレットはチーム設定で定義されます。
2. [webhook を作成]({{< relref path="#create-a-webhook" lang="ja" >}})し、エンドポイントと承認の詳細を定義し、必要なシークレットにアクセスするためのインテグレーションのアクセス権を付与します。
3. [オートメーションを作成]({{< relref path="#create-an-automation" lang="ja" >}})し、監視する[event]({{< relref path="/guides/core/automations/automation-events.md" lang="ja" >}})と W&B が送信するペイロードを定義します。ペイロードのために必要なシークレットに対して、オートメーションにアクセスを許可します。

## webhook の作成
チーム管理者は、チームに webhook を追加できます。

{{% alert %}}
webhook が Bearer トークンを必要とする場合、またはペイロードが機密文字列を必要とする場合は、webhook を作成する前にそれを含む[シークレットを作成]({{< relref path="/guides/core/secrets.md#add-a-secret" lang="ja" >}})してください。webhook には最大で1つのアクストークンと1つの他のシークレットを設定することができます。webhook の認証と承認の要件は、webhook のサービスによって決まります。
{{% /alert %}}

1. W&B にログインし、チーム設定ページに移動します。
2. **Webhooks** セクションで、**New webhook** をクリックします。
3. webhook に名前を提供します。
4. webhook のエンドポイント URL を提供します。
5. webhook が Bearer トークンを必要とする場合、**Access token** をそれを含む [secret]({{< relref path="/guides/core/secrets.md" lang="ja" >}})に設定します。webhook オートメーションを使用する際、W&B は `Authorization: Bearer` HTTP ヘッダーをアクストークンに設定し、`${ACCESS_TOKEN}` [payload variable]({{< relref path="#payload-variables" lang="ja" >}}) でトークンにアクセスできます。
6. webhook のペイロードにパスワードまたは他の機密文字列が必要な場合、**Secret** をその文字列を含むシークレットに設定します。webhook を使用するオートメーションを設定するとき、シークレットの名前に `$` を付けて [payload variable]({{< relref path="#payload-variables" lang="ja" >}}) としてシークレットにアクセスできます。

    webhook のアクセストークンがシークレットに保存されている場合は、アクセストークンとしてシークレットを指定するために次のステップを _必ず_ 完了してください。
7. W&B がエンドポイントに接続し、認証できることを確認するには：
    1. オプションで、テスト用のペイロードを提供します。ペイロード内で webhook がアクセス可能なシークレットを参照するには、その名前に `$` を付けます。このペイロードはテスト用であり保存されません。オートメーションのペイロードは [create the automation]({{< relref path="#create-a-webhook-automation" lang="ja" >}}) で設定します。シークレットとアクセストークンが `POST` リクエストで指定されている場所を表示するには、[Troubleshoot your webhook]({{< relref path="#troubleshoot-your-webhook" lang="ja" >}}) を参照してください。
    1. **Test** をクリックします。W&B は、設定された認証情報を使用して webhook のエンドポイントに接続しようとします。ペイロードを提供した場合は、W&B がそれを送信します。

    テストが成功しない場合は、webhook の設定を確認して再試行してください。必要に応じて、[Troubleshoot your webhook]({{< relref path="#troubleshoot-your-webhook" lang="ja" >}}) を参照してください。

これで webhook を使用する [オートメーションを作成する]({{< relref path="#create-a-webhook-automation" lang="ja" >}})ことができます。

## オートメーションの作成
[webhook を設定]({{< relref path="#reate-a-webhook" lang="ja" >}})した後、**Registry** または **Project** を選択し、webhook をトリガーするオートメーションを作成するための手順に従います。

{{< tabpane text=true >}}
{{% tab "Registry" %}}
レジストリ管理者は、そのレジストリ内でオートメーションを作成できます。レジストリのオートメーションは、将来追加されるものを含めて、そのレジストリ内のすべてのコレクションに適用されます。

1. W&B にログインします。
2. 詳細を確認するためにレジストリの名前をクリックします。
3. レジストリにスコープされているオートメーションを作成するには、**Automations** タブをクリックし、**Create automation** をクリックします。レジストリにスコープされているオートメーションは、そのすべてのコレクション（将来作成されるものを含む）に自動的に適用されます。

    レジストリ内の特定のコレクションのみにスコープされたオートメーションを作成するには、コレクションのアクション `...` メニューをクリックし、**Create automation** をクリックします。または、コレクションを表示しながら、コレクションの詳細ページの **Automations** セクションにある **Create automation** ボタンを使用してそれに対するオートメーションを作成します。
4. 監視する [**Event**]({{< relref path="/guides/core/automations/automation-events.md" lang="ja" >}}) を選択します。イベントによっては表示される追加フィールドを入力します。例えば、**An artifact alias is added** を選択した場合、**Alias regex** を指定する必要があります。**Next step** をクリックします。
5. [webhook]({{< relref path="#create-a-webhook" lang="ja" >}})を所有するチームを選択します。
6. **Action type** を **Webhooks** に設定し、使用する [webhook]({{< relref path="#create-a-webhook" lang="ja" >}}) を選択します。
7. webhook にアクセストークンを設定している場合、`${ACCESS_TOKEN}` [payload variable]({{< relref path="#payload-variables" lang="ja" >}}) でトークンにアクセスできます。webhook にシークレットを設定している場合、シークレットの名前に `$` を付けてペイロード内でアクセスできます。webhook の要件は webhook のサービスによって決まります。
8. **Next step** をクリックします。
9. オートメーションに名前を付けます。オプションで説明を入力します。**Create automation** をクリックします。

{{% /tab %}}
{{% tab "Project" %}}
W&B 管理者はプロジェクト内でオートメーションを作成できます。

1. W&B にログインし、プロジェクトページに移動します。
2. サイドバーの **Automations** をクリックします。
3. **Create automation** をクリックします。
4. 監視する [**Event**]({{< relref path="/guides/core/automations/automation-events.md" lang="ja" >}}) を選択します。

    1. 表示される、追加フィールドを入力します。例えば、**An artifact alias is added** を選択した場合、**Alias regex** を指定する必要があります。

    1. オプションでコレクションフィルタを指定します。それ以外の場合、オートメーションはプロジェクト内のすべてのコレクションに適用され、将来追加されるものも含まれます。
    
    **Next step** をクリックします。
5. [webhook]({{< relref path="#create-a-webhook" lang="ja" >}})を所有するチームを選択します。
6. **Action type** を **Webhooks** に設定し、使用する [webhook]({{< relref path="#create-a-webhook" lang="ja" >}}) を選択します。 
7. webhook がペイロードを必要とする場合、それを構築し、**Payload** フィールドに貼り付けます。webhook にアクセストークンを設定している場合、`${ACCESS_TOKEN}` [payload variable]({{< relref path="#payload-variables" lang="ja" >}}) でトークンにアクセスできます。webhook にシークレットを設定している場合、シークレットの名前に `$` を付けてペイロード内でアクセスできます。webhook の要件は webhook のサービスによって決まります。
8. **Next step** をクリックします。
9. オートメーションに名前を付けます。オプションで説明を入力します。**Create automation** をクリックします。

{{% /tab %}}
{{< /tabpane >}}

## オートメーションの表示と管理
{{< tabpane text=true >}}
{{% tab "Registry" %}}

- レジストリのオートメーションは、レジストリの **Automations** タブから管理します。
- コレクションのオートメーションは、コレクションの詳細ページの **Automations** セクションから管理します。

これらのページのいずれかから、レジストリ管理者は既存のオートメーションを管理できます。
- オートメーションの詳細を表示するには、その名前をクリックします。
- オートメーションを編集するには、そのアクションの `...` メニューをクリックし、**Edit automation** をクリックします。
- オートメーションを削除するには、そのアクションの `...` メニューをクリックし、**Delete automation** をクリックします。確認が必要です。

{{% /tab %}}
{{% tab "Project" %}}
W&B 管理者はプロジェクトの **Automations** タブからプロジェクトのオートメーションを表示および管理できます。

- オートメーションの詳細を表示するには、その名前をクリックします。
- オートメーションを編集するには、そのアクションの `...` メニューをクリックし、**Edit automation** をクリックします。
- オートメーションを削除するには、そのアクションの `...` メニューをクリックし、**Delete automation** をクリックします。確認が必要です。
{{% /tab %}}
{{< /tabpane >}}

## ペイロードのリファレンス
以下のセクションを使用して、webhook のペイロードを構築します。webhook とそのペイロードのテストについての詳細は、[Troubleshoot your webhook]({{< relref path="#troubleshoot-your-webhook" lang="ja" >}}) を参照してください。

### ペイロード変数
このセクションでは、webhook のペイロードを構築するために使用できる変数について説明します。

| Variable | Details |
|----------|---------|
| `${project_name}`             | アクションをトリガーした変更を所有するプロジェクトの名前。 |
| `${entity_name}`              | アクションをトリガーした変更を所有する entity またはチームの名前。 |
| `${event_type}`               | アクションをトリガーしたイベントのタイプ。 |
| `${event_author}`             | アクションをトリガーしたユーザー。 |
| `${artifact_collection_name}` | アーティファクトバージョンがリンクされているアーティファクトコレクションの名前。 |
| `${artifact_metadata.<KEY>}`  | アクションをトリガーしたアーティファクトバージョンのトップレベルのメタデータキーの任意の値。`<KEY>` をトップレベルのメタデータキーの名前に置き換えます。webhook のペイロードにはトップレベルのメタデータキーのみが利用可能です。 |
| `${artifact_version}`         | アクションをトリガーしたアーティファクトバージョンの [`Wandb.Artifact`]({{< relref path="/ref/python/artifact/" lang="ja" >}}) 表現。 |
| `${artifact_version_string}` | アクションをトリガーしたアーティファクトバージョンの`string` 表現。 |
| `${ACCESS_TOKEN}` | アクストークンが設定されている場合、[webhook]({{< relref path="#create-a-webhook" lang="ja" >}})で設定されたアクセストークンの値。アクセストークンは自動的に `Authorization: Bearer` HTTP ヘッダーに渡されます。 |
| `${SECRET_NAME}` | 設定されている場合、[webhook]({{< relref path="#create-a-webhook" lang="ja" >}})に設定されたシークレットの値。`SECRET_NAME` をシークレットの名前に置き換えます。 |

### ペイロードの例
このセクションでは、一般的なユースケースのための webhook ペイロードの例を示します。例は [payload variables]({{< relref path="#payload-variables" lang="ja" >}}) をどのように使用するかを示します。

{{< tabpane text=true >}}
{{% tab header="GitHub repository dispatch" value="github" %}}

{{% alert %}}
GHA ワークフローをトリガーするために必要なセットのアクセス許可を持っていることを確認してください。詳細については、[これらの GitHub Docs を参照してください](https://docs.github.com/en/rest/repos/repos?#create-a-repository-dispatch-event)。 
{{% /alert %}}

W&B からリポジトリディスパッチを送信して GitHub アクションをトリガーします。例えば、リポジトリディスパッチを `on` キーのトリガーとして受け入れる GitHub ワークフローファイルを持っているとしましょう。

```yaml
on:
repository_dispatch:
  types: BUILD_AND_DEPLOY
```

リポジトリ用のペイロードは次のようなものになるかもしれません。

```json
{
  "event_type": "BUILD_AND_DEPLOY",
  "client_payload": 
  {
    "event_author": "${event_author}",
    "artifact_version": "${artifact_version}",
    "artifact_version_string": "${artifact_version_string}",
    "artifact_collection_name": "${artifact_collection_name}",
    "project_name": "${project_name}",
    "entity_name": "${entity_name}"
    }
}
```

{{% alert %}}
webhook ペイロードの `event_type` キーは GitHub ワークフローファイルの `types` フィールドと一致しなければなりません。
{{% /alert %}}

レンダリングされたテンプレート文字列の内容と位置は、オートメーションが設定されているイベントまたはモデルバージョンによって異なります。`${event_type}` は `LINK_ARTIFACT` または `ADD_ARTIFACT_ALIAS` としてレンダリングされます。以下に例のマッピングを示します。

```text
${event_type} --> "LINK_ARTIFACT" または "ADD_ARTIFACT_ALIAS"
${event_author} --> "<wandb-user>"
${artifact_version} --> "wandb-artifact://_id/QXJ0aWZhY3Q6NTE3ODg5ODg3""
${artifact_version_string} --> "<entity>/model-registry/<registered_model_name>:<alias>"
${artifact_collection_name} --> "<registered_model_name>"
${project_name} --> "model-registry"
${entity_name} --> "<entity>"
```

テンプレート文字列を使用して W&B から GitHub Actions や他のツールにコンテキストを動的に渡します。これらのツールが Python スクリプトを呼び出すことができる場合、それらは [W&B API]({{< relref path="/guides/core/artifacts/download-and-use-an-artifact.md" lang="ja" >}})を通じて登録されたモデルアーティファクトを使用することができます。

- リポジトリディスパッチの詳細については、[GitHub Marketplace の公式ドキュメント](https://github.com/marketplace/actions/repository-dispatch)を参照してください。

- [Webhook Automations for Model Evaluation](https://www.youtube.com/watch?v=7j-Mtbo-E74&ab_channel=Weights%26Biases) と [Webhook Automations for Model Deployment](https://www.youtube.com/watch?v=g5UiAFjM2nA&ab_channel=Weights%26Biases) のビデオを視聴し、モデルの評価とデプロイメントのためのオートメーションを作成する方法を学びましょう。

- W&B の [レポート](https://wandb.ai/wandb/wandb-model-cicd/reports/Model-CI-CD-with-W-B--Vmlldzo0OTcwNDQw) をレビューし、GitHub Actions webhook オートメーションを使用した Model CI の作成方法を説明しています。この [GitHub リポジトリ](https://github.com/hamelsmu/wandb-modal-webhook) をチェックして、Modal Labs webhook を使用した model CI の作成方法を学びましょう。

{{% /tab %}}

{{% tab header="Microsoft Teams notification" value="microsoft"%}}

この例のペイロードは、webhook を使用して Teams チャンネルに通知する方法を示しています。

```json 
{
"@type": "MessageCard",
"@context": "http://schema.org/extensions",
"summary": "New Notification",
"sections": [
  {
    "activityTitle": "Notification from WANDB",
    "text": "This is an example message sent via Teams webhook.",
    "facts": [
      {
        "name": "Author",
        "value": "${event_author}"
      },
      {
        "name": "Event Type",
        "value": "${event_type}"
      }
    ],
    "markdown": true
  }
]
}
```

実行時に W&B データをペイロードに挿入するためにテンプレート文字列を使用できます（上記の Teams の例に示したように）。

{{% /tab %}}

{{% tab header="Slack notifications" value="slack"%}}

{{% alert %}}
このセクションは歴史的な目的で提供されます。現在、webhook を使用して Slack と統合している場合は、[新しい Slack インテグレーション]({{ relref "#create-a-slack-automation"}}) を使用するように設定を更新することをお勧めします。
{{% /alert %}}

Slack アプリをセットアップし、[Slack API ドキュメント](https://api.slack.com/messaging/webhooks)で強調されている指示に従って、着信 webhook インテグレーションを追加します。`Bot User OAuth Token` の下で指定されているシークレットが W&B webhook のアクストークンであることを確認してください。

以下はペイロードの例です。

```json
{
    "text": "New alert from WANDB!",
"blocks": [
    {
            "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": "Registry event: ${event_type}"
        }
    },
        {
            "type":"section",
            "text": {
            "type": "mrkdwn",
            "text": "New version: ${artifact_version_string}"
        }
        },
        {
        "type": "divider"
    },
        {
            "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": "Author: ${event_author}"
        }
        }
    ]
}
```

{{% /tab %}}
{{< /tabpane >}}

## webhook のトラブルシューティング
W&B アプリ UI または Bash スクリプトを使用して、インタラクティブに webhook のトラブルシューティングを行います。新しい webhook を作成する際や既存の webhook を編集する際に webhook をトラブルシューティングできます。

{{< tabpane text=true >}}
{{% tab header="W&B App UI" value="app" %}}

チーム管理者は W&B アプリ UI を使用して webhook をインタラクティブにテストできます。

1. W&B チーム設定ページに移動します。
2. **Webhooks** セクションまでスクロールします。
3. webhook の名前の横にある三点リーダー（ミートボールアイコン）をクリックします。
4. **Test** を選択します。
5. 現れた UI パネルから、表示されるフィールドに POST リクエストを貼り付けます。 
    {{< img src="/images/models/webhook_ui.png" alt="webhook ペイロードのテストデモ" >}}
6. **Test webhook** をクリックします。W&B アプリ UI 内で、W&B はエンドポイントからの応答を投稿します。
    {{< img src="/images/models/webhook_ui_testing.gif" alt="webhook のテストデモ" >}}

[Testing Webhooks in Weights & Biases](https://www.youtube.com/watch?v=bl44fDpMGJw&ab_channel=Weights%26Biases) のビデオを見て、デモをご覧ください。
{{% /tab %}}

{{% tab header="Bash script" value="bash"%}}

このシェルスクリプトは、W&B が webhook オートメーションに送信する `POST` リクエストを生成する1つの方法を示しています。

以下のコードをシェルスクリプトにコピーし、webhook のトラブルシューティングを行います。以下の値を指定してください。

* `ACCESS_TOKEN`
* `SECRET`
* `PAYLOAD`
* `API_ENDPOINT`

{{< prism file="/webhook_test.sh" title="webhook_test.sh">}}{{< /prism >}}

{{% /tab %}}
{{< /tabpane >}}