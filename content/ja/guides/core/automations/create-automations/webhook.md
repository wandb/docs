---
title: Webhook の自動化を作成
menu:
  default:
    identifier: ja-guides-core-automations-create-automations-webhook
    parent: automations
weight: 3
---

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

このページでは、webhook オートメーションを作成する方法について説明します。Slack オートメーションを作成するには、代わりに [Slack オートメーションを作成する]({{< relref path="/guides/core/automations/create-automations/slack.md" lang="ja" >}}) を参照してください。

大まかに言うと、webhook オートメーションを作成するには、次の手順を実行します。
1. 必要に応じて、アクセストークン、パスワード、SSH キーなど、オートメーションで必要となる機密性の高い文字列ごとに [W&B シークレットを作成します]({{< relref path="/guides/core/secrets.md" lang="ja" >}})。シークレットは、**Team Settings** で定義されます。
1. [webhook を作成し]({{< relref path="#create-a-webhook" lang="ja" >}})、エンドポイントと認証の詳細を定義し、必要なシークレットへのインテグレーション アクセスを許可します。
1. [オートメーションを作成し]({{< relref path="#create-an-automation" lang="ja" >}})、監視する [イベント]({{< relref path="/guides/core/automations/automation-events.md" lang="ja" >}}) と W&B が送信するペイロードを定義します。ペイロードに必要なシークレットへのオートメーション アクセスを許可します。

## webhook を作成する
Team 管理者は、Team の webhook を追加できます。

{{% alert %}}
webhook にベアラートークンが必要な場合、またはペイロードに機密性の高い文字列が必要な場合は、webhook を作成する前に [それを含むシークレットを作成します]({{< relref path="/guides/core/secrets.md#add-a-secret" lang="ja" >}})。webhook に設定できるアクセストークンは最大 1 つ、その他のシークレットは最大 1 つです。webhook の認証および認可要件は、webhook のサービスによって決定されます。
{{% /alert %}}

1. W&B にログインし、**Team Settings** ページに移動します。
1. **Webhooks** セクションで、**New webhook** をクリックします。
1. webhook の名前を入力します。
1. webhook のエンドポイント URL を入力します。
1. webhook にベアラートークンが必要な場合は、**Access token** を、それを含む [シークレット]({{< relref path="/guides/core/secrets.md" lang="ja" >}}) に設定します。webhook オートメーションを使用すると、W&B は `Authorization: Bearer` HTTP ヘッダーをアクセストークンに設定し、`${ACCESS_TOKEN}` [ペイロード変数]({{< relref path="#payload-variables" lang="ja" >}}) でトークンにアクセスできます。W&B が webhook サービスに送信する `POST` リクエストの構造の詳細については、[webhook のトラブルシューティング]({{< relref path="#troubleshoot-your-webhook" lang="ja" >}}) を参照してください。
1. webhook のペイロードにパスワードまたはその他の機密性の高い文字列が必要な場合は、**Secret** を、それを含むシークレットに設定します。webhook を使用するオートメーションを設定する際、シークレットの名前の前に `$` を付けることで、[ペイロード変数]({{< relref path="#payload-variables" lang="ja" >}}) としてシークレットにアクセスできます。

    webhook のアクセストークンがシークレットに保存されている場合は、アクセストークンとしてシークレットを指定するために、次のステップも完了する必要があります。
1. W&B がエンドポイントに接続して認証できることを確認するには:
    1. 必要に応じて、テスト用のペイロードを提供します。ペイロード内で webhook がアクセスできるシークレットを参照するには、その名前の前に `$` を付けます。このペイロードはテスト専用であり、保存されません。オートメーションのペイロードは、[オートメーションを作成する]({{< relref path="#create-a-webhook-automation" lang="ja" >}}) 際に設定します。`POST` リクエストでシークレットとアクセストークンがどこに指定されているかを確認するには、[webhook のトラブルシューティング]({{< relref path="#troubleshoot-your-webhook" lang="ja" >}}) を参照してください。
    1. **Test** をクリックします。W&B は、設定した資格情報を使用して webhook のエンドポイントへの接続を試みます。ペイロードを提供した場合、W&B はそれを送信します。

    テストが成功しない場合は、webhook の設定を確認して再試行してください。必要に応じて、[webhook のトラブルシューティング]({{< relref path="#troubleshoot-your-webhook" lang="ja" >}}) を参照してください。

![Team 内の 2 つの webhook を示すスクリーンショット](/images/automations/webhooks.png)

これで、webhook を使用する [オートメーションを作成できます]({{< relref path="#create-a-webhook-automation" lang="ja" >}})。

## オートメーションを作成する
[webhook を設定したら]({{< relref path="#create-a-webhook" lang="ja" >}})、**Registry** または **Project** を選択し、これらの手順に従って webhook をトリガーするオートメーションを作成します。

{{< tabpane text=true >}}
{{% tab "Registry" %}}
Registry 管理者は、その Registry 内にオートメーションを作成できます。Registry オートメーションは、将来追加されるものを含め、Registry 内のすべてのコレクションに適用されます。

1. W&B にログインします。
1. Registry の名前をクリックして、その詳細を表示します。
1. Registry にスコープされたオートメーションを作成するには、**Automations** タブをクリックし、**Create automation** をクリックします。Registry にスコープされたオートメーションは、そのすべてのコレクション (将来作成されるものを含む) に自動的に適用されます。

    Registry 内の特定のコレクションにのみスコープされたオートメーションを作成するには、コレクションのアクション `...` メニューをクリックし、**Create automation** をクリックします。あるいは、コレクションを表示中に、コレクションの詳細ページの **Automations** セクションにある **Create automation** ボタンを使用してオートメーションを作成します。
1. 監視する [イベント]({{< relref path="/guides/core/automations/automation-events.md" lang="ja" >}}) を選択します。表示される追加フィールドに記入します。これらはイベントによって異なります。たとえば、**An artifact alias is added** を選択した場合は、**Alias regex** を指定する必要があります。**Next step** をクリックします。
1. [webhook]({{< relref path="#create-a-webhook" lang="ja" >}}) を所有する Team を選択します。
1. **Action type** を **Webhooks** に設定し、使用する [webhook]({{< relref path="#create-a-webhook" lang="ja" >}}) を選択します。
1. webhook のアクセストークンを設定した場合、`${ACCESS_TOKEN}` [ペイロード変数]({{< relref path="#payload-variables" lang="ja" >}}) でトークンにアクセスできます。webhook のシークレットを設定した場合、その名前の前に `$` を付けることでペイロード内のシークレットにアクセスできます。webhook の要件は、webhook のサービスによって決定されます。
1. **Next step** をクリックします。
1. オートメーションの名前を入力します。必要に応じて、説明を入力します。**Create automation** をクリックします。

{{% /tab %}}
{{% tab "Project" %}}
W&B 管理者は、Project 内にオートメーションを作成できます。

1. W&B にログインし、Project ページに移動します。
1. サイドバーで **Automations** をクリックし、**Create automation** をクリックします。

    または、Workspace の折れ線グラフから、表示されているメトリックの [run メトリック オートメーション]({{< relref path="/guides/core/automations/automation-events.md#run-events" lang="ja" >}}) をすばやく作成できます。パネルにカーソルを合わせ、パネルの上部にあるベルアイコンをクリックします。
    {{< img src="/images/automations/run_metric_automation_from_panel.png" alt="オートメーションのベルアイコンの場所" >}}
1. Artifact のエイリアスが追加されたとき、または run メトリックが指定されたしきい値に達したときなど、監視する [イベント]({{< relref path="/guides/core/automations/automation-events.md" lang="ja" >}}) を選択します。

    1. 表示される追加フィールドに記入します。これらはイベントによって異なります。たとえば、**An artifact alias is added** を選択した場合は、**Alias regex** を指定する必要があります。

    1. 必要に応じてコレクションフィルターを指定します。指定しない場合、オートメーションは将来追加されるものを含め、Project 内のすべてのコレクションに適用されます。

    **Next step** をクリックします。
1. [webhook]({{< relref path="#create-a-webhook" lang="ja" >}}) を所有する Team を選択します。
1. **Action type** を **Webhooks** に設定し、使用する [webhook]({{< relref path="#create-a-webhook" lang="ja" >}}) を選択します。
1. webhook にペイロードが必要な場合は、それを構築し、**Payload** フィールドに貼り付けます。webhook のアクセストークンを設定した場合、`${ACCESS_TOKEN}` [ペイロード変数]({{< relref path="#payload-variables" lang="ja" >}}) でトークンにアクセスできます。webhook のシークレットを設定した場合、その名前の前に `$` を付けることでペイロード内のシークレットにアクセスできます。webhook の要件は、webhook のサービスによって決定されます。
1. **Next step** をクリックします。
1. オートメーションの名前を入力します。必要に応じて、説明を入力します。**Create automation** をクリックします。

{{% /tab %}}
{{< /tabpane >}}

## オートメーションを表示および管理する
{{< tabpane text=true >}}
{{% tab "Registry" %}}

- Registry のオートメーションは、Registry の **Automations** タブから管理します。
- コレクションのオートメーションは、コレクションの詳細ページの **Automations** セクションから管理します。

これらのページのいずれかから、Registry 管理者は既存のオートメーションを管理できます。
- オートメーションの詳細を表示するには、その名前をクリックします。
- オートメーションを編集するには、そのアクション `...` メニューをクリックし、**Edit automation** をクリックします。
- オートメーションを削除するには、そのアクション `...` メニューをクリックし、**Delete automation** をクリックします。確認が必要です。

{{% /tab %}}
{{% tab "Project" %}}
W&B 管理者は、Project の **Automations** タブから Project のオートメーションを表示および管理できます。

- オートメーションの詳細を表示するには、その名前をクリックします。
- オートメーションを編集するには、そのアクション `...` メニューをクリックし、**Edit automation** をクリックします。
- オートメーションを削除するには、そのアクション `...` メニューをクリックし、**Delete automation** をクリックします。確認が必要です。
{{% /tab %}}
{{< /tabpane >}}

## ペイロード参照
これらのセクションを使用して、webhook のペイロードを構築します。webhook とそのペイロードのテストの詳細については、[webhook のトラブルシューティング]({{< relref path="#troubleshoot-your-webhook" lang="ja" >}}) を参照してください。

### ペイロード変数
このセクションでは、webhook のペイロードを構築するために使用できる変数について説明します。

| Variable | Details |
|----------|---------|
| `${project_name}`             | アクションをトリガーした変更を所有する Project の名前です。 |
| `${entity_name}`              | アクションをトリガーした変更を所有する Entity または Team の名前です。 |
| `${event_type}`               | アクションをトリガーしたイベントのタイプです。 |
| `${event_author}`             | アクションをトリガーした User です。 |
| `${alias}`                    | オートメーションが **An artifact alias is added** イベントによってトリガーされた場合、Artifact のエイリアスを含みます。その他のオートメーションの場合、この変数は空白です。 |
| `${tag}`                      | オートメーションが **An artifact tag is added** イベントによってトリガーされた場合、Artifact のタグを含みます。その他のオートメーションの場合、この変数は空白です。 |
| `${artifact_collection_name}` | Artifact のバージョンがリンクされている Artifact コレクションの名前です。 |
| `${artifact_metadata.<KEY>}`  | アクションをトリガーした Artifact のバージョンからの任意のトップレベルのメタデータキーの値です。`<KEY>` をトップレベルのメタデータキーの名前に置き換えます。トップレベルのメタデータキーのみが webhook のペイロードで利用可能です。 |
| `${artifact_version}`         | アクションをトリガーした Artifact のバージョンの [`Wandb.Artifact`]({{< relref path="/ref/python/sdk/classes/artifact.md/" lang="ja" >}}) 表現です。 |
| `${artifact_version_string}` | アクションをトリガーした Artifact のバージョンの `string` 表現です。 |
| `${ACCESS_TOKEN}` | アクセストークンが設定されている場合、[webhook]({{< relref path="#create-a-webhook" lang="ja" >}}) に設定されたアクセストークンの値です。アクセストークンは、`Authorization: Bearer` HTTP ヘッダーで自動的に渡されます。 |
| `${SECRET_NAME}` | 設定されている場合、[webhook]({{< relref path="#create-a-webhook" lang="ja" >}}) に設定されたシークレットの値です。`SECRET_NAME` をシークレットの名前に置き換えます。 |

### ペイロードの例
このセクションには、いくつかの一般的なユースケースの webhook ペイロードの例が含まれています。これらの例は、[ペイロード変数]({{< relref path="#payload-variables" lang="ja" >}}) の使用方法を示しています。

{{< tabpane text=true >}}
{{% tab header="GitHub repository dispatch" value="github" %}}

{{% alert %}}
アクセストークンが GHA ワークフローをトリガーするために必要な権限セットを持っていることを確認してください。詳細については、[GitHub Docs を参照してください](https://docs.github.com/en/rest/repos/repos?#create-a-repository-dispatch-event)。
{{% /alert %}}

W&B からリポジトリディスパッチを送信して、GitHub アクションをトリガーします。たとえば、`on` キーのトリガーとしてリポジトリディスパッチを受け入れる GitHub ワークフローファイルがあるとします。

```yaml
on:
repository_dispatch:
  types: BUILD_AND_DEPLOY
```

リポジトリのペイロードは次のようになります。

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
webhook ペイロードの `event_type` キーは、GitHub ワークフロー YAML ファイルの `types` フィールドと一致する必要があります。
{{% /alert %}}

レンダリングされたテンプレート文字列の内容と配置は、オートメーションが設定されているイベントまたはモデル バージョンによって異なります。`${event_type}` は `LINK_ARTIFACT` または `ADD_ARTIFACT_ALIAS` のいずれかとしてレンダリングされます。以下のマッピング例を参照してください。

```text
${event_type} --> "LINK_ARTIFACT" or "ADD_ARTIFACT_ALIAS"
${event_author} --> "<wandb-user>"
${artifact_version} --> "wandb-artifact://_id/QXJ0aWZhY3Q6NTE3ODg5ODg3""
${artifact_version_string} --> "<entity>/model-registry/<registered_model_name>:<alias>"
${artifact_collection_name} --> "<registered_model_name>"
${project_name} --> "model-registry"
${entity_name} --> "<entity>"
```

テンプレート文字列を使用して、W&B から GitHub Actions やその他のツールにコンテキストを動的に渡します。これらのツールが Python スクリプトを呼び出すことができる場合、[W&B API]({{< relref path="/guides/core/artifacts/download-and-use-an-artifact.md" lang="ja" >}}) を介して Registered Model Artifacts を使用できます。

- リポジトリディスパッチの詳細については、[GitHub Marketplace の公式ドキュメント](https://github.com/marketplace/actions/repository-dispatch) を参照してください。

- [モデル評価のための Webhook オートメーション](https://www.youtube.com/watch?v=7j-Mtbo-E74&ab_channel=Weights%26Biases) と [モデルデプロイメントのための Webhook オートメーション](https://www.youtube.com/watch?v=g5UiAFjM2nA&ab_channel=Weights%26Biases) のビデオを見て、モデル評価とデプロイメントのためのオートメーションを作成する方法を案内してください。

- W&B の [Reports](https://wandb.ai/wandb/wandb-model-cicd/reports/Model-CI-CD-with-W-B--Vmlldzo0OTcwNDQw) を確認してください。これは、モデル CI に GitHub Actions webhook オートメーションを使用する方法を示しています。Modal Labs webhook でモデル CI を作成する方法については、この [GitHub リポジトリ](https://github.com/hamelsmu/wandb-modal-webhook) を確認してください。

{{% /tab %}}

{{% tab header="Microsoft Teams notification" value="microsoft"%}}

このペイロードの例は、webhook を使用して Teams チャンネルに通知する方法を示しています。

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

上記の Teams の例に示すように、テンプレート文字列を使用して、実行時に W&B データをペイロードに注入できます。

{{% /tab %}}

{{% tab header="Slack notifications" value="slack"%}}

{{% alert %}}
このセクションは歴史的な目的で提供されています。現在 webhook を使用して Slack と統合している場合は、[新しい Slack インテグレーション]({{ relref "#create-a-slack-automation"}}) を使用するように設定を更新することをお勧めします。
{{% /alert %}}

[Slack API ドキュメント](https://api.slack.com/messaging/webhooks) で強調されている指示に従って、Slack アプリを設定し、受信 webhook インテグレーションを追加します。`Bot User OAuth Token` の下に指定されたシークレットが、W&B webhook のアクセストークンであることを確認してください。

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
W&B App UI でインタラクティブに、または Bash スクリプトでプログラム的に webhook のトラブルシューティングを行うことができます。新しい webhook を作成するとき、または既存の webhook を編集するときに webhook のトラブルシューティングを行うことができます。

W&B が `POST` リクエストに使用する形式の詳細については、**Bash script** タブを参照してください。

{{< tabpane text=true >}}
{{% tab header="W&B App UI" value="app" %}}

Team 管理者は、W&B App UI を使用して webhook をインタラクティブにテストできます。

1. W&B Team Settings ページに移動します。
2. **Webhooks** セクションまでスクロールします。
3. webhook 名の横にある横 3 点リーダー (ミートボールアイコン) をクリックします。
4. **Test** を選択します。
5. 表示される UI パネルから、表示されるフィールドに POST リクエストを貼り付けます。
    {{< img src="/images/models/webhook_ui.png" alt="webhook ペイロードのテストデモ" >}}
6. **Test webhook** をクリックします。W&B App UI 内で、W&B はエンドポイントからの応答を投稿します。
    {{< img src="/images/models/webhook_ui_testing.gif" alt="webhook のテストデモ" >}}

デモンストレーションについては、ビデオ [W&B での Webhook のテスト](https://www.youtube.com/watch?v=bl44fDpMGJw&ab_channel=Weights%26Biases) をご覧ください。
{{% /tab %}}

{{% tab header="Bash script" value="bash"%}}

このシェルスクリプトは、W&B が webhook オートメーションがトリガーされたときに送信するリクエストと同様の `POST` リクエストを生成する 1 つのメソッドを示しています。

以下のコードをシェルスクリプトにコピー＆ペーストして、webhook のトラブルシューティングを行ってください。以下の値をご自身の値に指定してください:

* `ACCESS_TOKEN`
* `SECRET`
* `PAYLOAD`
* `API_ENDPOINT`

{{< prism file="/webhook_test.sh" title="webhook_test.sh">}}{{< /prism >}}

{{% /tab %}}
{{< /tabpane >}}