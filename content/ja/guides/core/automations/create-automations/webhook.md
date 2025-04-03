---
title: Create a webhook automation
menu:
  default:
    identifier: ja-guides-core-automations-create-automations-webhook
    parent: automations
weight: 3
---

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

このページでは、webhook [ オートメーション ]({{< relref path="/guides/core/automations/" lang="ja" >}}) の作成方法について説明します。Slack オートメーションを作成するには、代わりに [Slack オートメーションの作成]({{< relref path="/guides/core/automations/create-automations/slack.md" lang="ja" >}}) を参照してください。

大まかに言うと、webhook オートメーションを作成するには、次の手順を実行します。
1. 必要に応じて、アクセストークン、パスワード、SSH キーなど、オートメーションに必要な機密文字列ごとに [W&B シークレットを作成]({{< relref path="/guides/core/secrets.md" lang="ja" >}}) します。シークレットは、 Team の設定で定義されます。
1. [webhook を作成]({{< relref path="#create-a-webhook" lang="ja" >}}) して、エンドポイントと認証の詳細を定義し、インテグレーションに必要なシークレットへのアクセスを許可します。
1. [オートメーションを作成]({{< relref path="#create-an-automation" lang="ja" >}}) して、監視する [ イベント ]({{< relref path="/guides/core/automations/automation-events.md" lang="ja" >}}) と、Weights & Biases が送信するペイロードを定義します。ペイロードに必要なシークレットへのオートメーションアクセスを許可します。

## webhook を作成する
Team の管理者は、 Team に webhook を追加できます。

{{% alert %}}
webhook に Bearer トークンが必要な場合、またはそのペイロードに機密文字列が必要な場合は、webhook を作成する前に [それを含むシークレットを作成]({{< relref path="/guides/core/secrets.md#add-a-secret" lang="ja" >}}) してください。webhook に対して設定できるアクセストークンとその他のシークレットは、それぞれ最大 1 つです。webhook の認証および認可の要件は、webhook のサービスによって決まります。
{{% /alert %}}

1. Weights & Biases にログインし、Team の Settings ページに移動します。
1. **Webhooks** セクションで、**New webhook** をクリックします。
1. webhook の名前を入力します。
1. webhook のエンドポイント URL を入力します。
1. webhook に Bearer トークンが必要な場合は、**Access token** をそれを含む [シークレット]({{< relref path="/guides/core/secrets.md" lang="ja" >}}) に設定します。webhook オートメーションを使用すると、Weights & Biases は `Authorization: Bearer` HTTP ヘッダーをアクセストークンに設定し、`${ACCESS_TOKEN}` [ペイロード変数]({{< relref path="#payload-variables" lang="ja" >}}) でトークンにアクセスできます。
1. webhook のペイロードにパスワードまたはその他の機密文字列が必要な場合は、**Secret** をそれを含むシークレットに設定します。webhook を使用するオートメーションを設定すると、名前の先頭に `$` を付けることで、[ペイロード変数]({{< relref path="#payload-variables" lang="ja" >}}) としてシークレットにアクセスできます。

    webhook のアクセストークンがシークレットに保存されている場合は、次の手順も完了して、シークレットをアクセストークンとして指定する _必要_ があります。
1. Weights & Biases がエンドポイントに接続して認証できることを確認するには：
    1. 必要に応じて、テストするペイロードを指定します。ペイロードで webhook がアクセスできるシークレットを参照するには、名前の先頭に `$` を付けます。このペイロードはテスト専用であり、保存されません。オートメーションのペイロードは、[オートメーションを作成]({{< relref path="#create-a-webhook-automation" lang="ja" >}}) するときに設定します。シークレットとアクセストークンが `POST` リクエストのどこに指定されているかを確認するには、[webhook のトラブルシューティング]({{< relref path="#troubleshoot-your-webhook" lang="ja" >}}) を参照してください。
    1. **Test** をクリックします。Weights & Biases は、設定した資格情報を使用して webhook のエンドポイントへの接続を試みます。ペイロードを指定した場合は、Weights & Biases がそれを送信します。

    テストが成功しない場合は、webhook の設定を確認して、もう一度お試しください。必要に応じて、[webhook のトラブルシューティング]({{< relref path="#troubleshoot-your-webhook" lang="ja" >}}) を参照してください。

これで、webhook を使用する [オートメーションを作成]({{< relref path="#create-a-webhook-automation" lang="ja" >}}) できます。

## オートメーションを作成する
[webhook を設定]({{< relref path="#reate-a-webhook" lang="ja" >}}) したら、**Registry** または **Project** を選択し、次の手順に従って webhook をトリガーするオートメーションを作成します。

{{< tabpane text=true >}}
{{% tab "Registry" %}}
Registry 管理者は、その Registry でオートメーションを作成できます。Registry オートメーションは、今後追加されるものを含め、Registry 内のすべてのコレクションに適用されます。

1. Weights & Biases にログインします。
1. Registry の名前をクリックして詳細を表示します。
1. Registry の範囲に設定されたオートメーションを作成するには、**Automations** タブをクリックし、**Create automation** をクリックします。Registry の範囲に設定されたオートメーションは、そのすべてのコレクション（今後作成されるものを含む）に自動的に適用されます。

    Registry 内の特定のコレクションのみを範囲とするオートメーションを作成するには、コレクションのアクション `...` メニューをクリックし、**Create automation** をクリックします。または、コレクションを表示しながら、コレクションの詳細ページの **Automations** セクションにある **Create automation** ボタンを使用して、コレクションのオートメーションを作成します。
1. 監視する [**Event**]({{< relref path="/guides/core/automations/automation-events.md" lang="ja" >}}) を選択します。イベントに応じて表示される追加フィールドに入力します。たとえば、**An artifact alias is added** を選択した場合は、**Alias regex** を指定する必要があります。**Next step** をクリックします。
1. [webhook]({{< relref path="#create-a-webhook" lang="ja" >}}) を所有する Team を選択します。
1. **Action type** を **Webhooks** に設定し、使用する [webhook]({{< relref path="#create-a-webhook" lang="ja" >}}) を選択します。
1. webhook のアクセストークンを設定した場合は、`${ACCESS_TOKEN}` [ペイロード変数]({{< relref path="#payload-variables" lang="ja" >}}) でトークンにアクセスできます。webhook のシークレットを設定した場合は、名前の先頭に `$` を付けることで、ペイロードでアクセスできます。webhook の要件は、webhook のサービスによって決まります。
1. **Next step** をクリックします。
1. オートメーションの名前を入力します。必要に応じて、説明を入力します。**Create automation** をクリックします。

{{% /tab %}}
{{% tab "Project" %}}
Weights & Biases 管理者は、Project でオートメーションを作成できます。

1. Weights & Biases にログインし、Project ページに移動します。
1. サイドバーで、**Automations** をクリックします。
1. **Create automation** をクリックします。
1. 監視する [**Event**]({{< relref path="/guides/core/automations/automation-events.md" lang="ja" >}}) を選択します。

    1. イベントに応じて表示される追加フィールドに入力します。たとえば、**An artifact alias is added** を選択した場合は、**Alias regex** を指定する必要があります。

    1. 必要に応じて、コレクションフィルターを指定します。それ以外の場合、オートメーションは、今後追加されるものを含め、Project 内のすべてのコレクションに適用されます。
    
    **Next step** をクリックします。
1. [webhook]({{< relref path="#create-a-webhook" lang="ja" >}}) を所有する Team を選択します。
1. **Action type** を **Webhooks** に設定し、使用する [webhook]({{< relref path="#create-a-webhook" lang="ja" >}}) を選択します。
1. webhook にペイロードが必要な場合は、ペイロードを作成して **Payload** フィールドに貼り付けます。webhook のアクセストークンを設定した場合は、`${ACCESS_TOKEN}` [ペイロード変数]({{< relref path="#payload-variables" lang="ja" >}}) でトークンにアクセスできます。webhook のシークレットを設定した場合は、名前の先頭に `$` を付けることで、ペイロードでアクセスできます。webhook の要件は、webhook のサービスによって決まります。
1. **Next step** をクリックします。
1. オートメーションの名前を入力します。必要に応じて、説明を入力します。**Create automation** をクリックします。

{{% /tab %}}
{{< /tabpane >}}

## オートメーションの表示と管理
{{< tabpane text=true >}}
{{% tab "Registry" %}}

- Registry のオートメーションは、Registry の **Automations** タブから管理します。
- コレクションのオートメーションは、コレクションの詳細ページの **Automations** セクションから管理します。

これらのページのいずれかから、Registry 管理者は既存のオートメーションを管理できます。
- オートメーションの詳細を表示するには、その名前をクリックします。
- オートメーションを編集するには、アクション `...` メニューをクリックし、**Edit automation** をクリックします。
- オートメーションを削除するには、アクション `...` メニューをクリックし、**Delete automation** をクリックします。確認が必要です。

{{% /tab %}}
{{% tab "Project" %}}
Weights & Biases 管理者は、Project の **Automations** タブから Project のオートメーションを表示および管理できます。

- オートメーションの詳細を表示するには、その名前をクリックします。
- オートメーションを編集するには、アクション `...` メニューをクリックし、**Edit automation** をクリックします。
- オートメーションを削除するには、アクション `...` メニューをクリックし、**Delete automation** をクリックします。確認が必要です。
{{% /tab %}}
{{< /tabpane >}}

## ペイロードリファレンス
これらのセクションを使用して、webhook のペイロードを作成します。webhook とそのペイロードのテストの詳細については、[webhook のトラブルシューティング]({{< relref path="#troubleshoot-your-webhook" lang="ja" >}}) を参照してください。

### ペイロード変数
このセクションでは、webhook のペイロードの作成に使用できる変数について説明します。

| 変数                      | 詳細                                                                                                                                                                      |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `${project_name}`             | アクションをトリガーした変更を所有する Project の名前。                                                                                                                              |
| `${entity_name}`              | アクションをトリガーした変更を所有する Entity または Team の名前。                                                                                                                               |
| `${event_type}`               | アクションをトリガーしたイベントのタイプ。                                                                                                                                  |
| `${event_author}`             | アクションをトリガーしたユーザー。                                                                                                                                      |
| `${artifact_collection_name}` | Artifact バージョンがリンクされている Artifact コレクションの名前。                                                                                                                     |
| `${artifact_metadata.<KEY>}`  | アクションをトリガーした Artifact バージョンからの任意のトップレベル メタデータ キーの値。`<KEY>` をトップレベル メタデータ キーの名前に置き換えます。トップレベル メタデータ キーのみが、webhook のペイロードで使用できます。 |
| `${artifact_version}`         | アクションをトリガーした Artifact バージョンの [`Wandb.Artifact`]({{< relref path="/ref/python/artifact/" lang="ja" >}}) 表現。                                                                                                               |
| `${artifact_version_string}` | アクションをトリガーした Artifact バージョンの `string` 表現。                                                                                                                            |
| `${ACCESS_TOKEN}` | アクセストークンが設定されている場合は、[webhook]({{< relref path="#create-a-webhook" lang="ja" >}}) で設定されたアクセストークンの値。アクセストークンは、`Authorization: Bearer` HTTP ヘッダーで自動的に渡されます。                                                                   |
| `${SECRET_NAME}` | 設定されている場合は、[webhook]({{< relref path="#create-a-webhook" lang="ja" >}}) で設定されたシークレットの値。`SECRET_NAME` をシークレットの名前に置き換えます。                                                                                                       |

### ペイロードの例
このセクションには、一般的なユースケースの webhook ペイロードの例が含まれています。この例では、[ペイロード変数]({{< relref path="#payload-variables" lang="ja" >}}) の使用方法を示します。

{{< tabpane text=true >}}
{{% tab header="GitHub リポジトリディスパッチ" value="github" %}}

{{% alert %}}
アクセストークンに、GHA ワークフローをトリガーするために必要な権限セットがあることを確認してください。詳細については、[GitHub ドキュメントを参照してください](https://docs.github.com/en/rest/repos/repos?#create-a-repository-dispatch-event)。
{{% /alert %}}

Weights & Biases からリポジトリディスパッチを送信して、GitHub アクションをトリガーします。たとえば、`on` キーのトリガーとしてリポジトリディスパッチを受け入れる GitHub ワークフローファイルがあるとします。

```yaml
on:
repository_dispatch:
  types: BUILD_AND_DEPLOY
```

リポジトリのペイロードは、次のようになります。

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

レンダリングされたテンプレート文字列の内容と配置は、オートメーションが設定されているイベントまたはモデルバージョンによって異なります。`${event_type}` は、`LINK_ARTIFACT` または `ADD_ARTIFACT_ALIAS` のいずれかとしてレンダリングされます。以下に、マッピングの例を示します。

```text
${event_type} --> "LINK_ARTIFACT" or "ADD_ARTIFACT_ALIAS"
${event_author} --> "<wandb-user>"
${artifact_version} --> "wandb-artifact://_id/QXJ0aWZhY3Q6NTE3ODg5ODg3""
${artifact_version_string} --> "<entity>/model-registry/<registered_model_name>:<alias>"
${artifact_collection_name} --> "<registered_model_name>"
${project_name} --> "model-registry"
${entity_name} --> "<entity>"
```

テンプレート文字列を使用して、Weights & Biases から GitHub Actions およびその他のツールにコンテキストを動的に渡します。これらのツールが Python スクリプトを呼び出すことができる場合は、[Weights & Biases API]({{< relref path="/guides/core/artifacts/download-and-use-an-artifact.md" lang="ja" >}}) を使用して、登録されたモデル Artifacts を利用できます。

- リポジトリディスパッチの詳細については、[GitHub Marketplace の公式ドキュメント](https://github.com/marketplace/actions/repository-dispatch) を参照してください。

- [モデル評価の Webhook オートメーション](https://www.youtube.com/watch?v=7j-Mtbo-E74&ab_channel=Weights%26Biases) および [モデルデプロイメントの Webhook オートメーション](https://www.youtube.com/watch?v=g5UiAFjM2nA&ab_channel=Weights%26Biases) のビデオをご覧ください。これらのビデオでは、モデル評価およびデプロイメントのオートメーションを作成する方法を説明しています。

- Weights & Biases [レポート](https://wandb.ai/wandb/wandb-model-cicd/reports/Model-CI-CD-with-W-B--Vmlldzo0OTcwNDQw) を確認してください。このレポートでは、モデル CI に Github Actions webhook オートメーションを使用する方法を示しています。この [GitHub リポジトリ](https://github.com/hamelsmu/wandb-modal-webhook) をチェックして、Modal Labs webhook でモデル CI を作成する方法を学んでください。

{{% /tab %}}

{{% tab header="Microsoft Teams 通知" value="microsoft"%}}

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

テンプレート文字列を使用して、実行時に Weights & Biases データをペイロードに挿入できます（上記の Teams の例に示すように）。

{{% /tab %}}

{{% tab header="Slack 通知" value="slack"%}}

{{% alert %}}
このセクションは、履歴を目的として提供されています。現在 webhook を使用して Slack と統合している場合は、[新しい Slack インテグレーション]({{ relref "#create-a-slack-automation"}}) を使用するように構成を更新することをお勧めします。
{{% /alert %}}

[Slack API ドキュメント](https://api.slack.com/messaging/webhooks) で強調表示されている手順に従って、Slack アプリを設定し、受信 webhook インテグレーションを追加します。`Bot User OAuth Token` で指定されたシークレットが、Weights & Biases webhook のアクセストークンであることを確認してください。

以下は、ペイロードの例です。

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
Weights & Biases App UI を使用してインタラクティブに、または Bash スクリプトを使用してプログラムで webhook のトラブルシューティングを行います。新しい webhook を作成するとき、または既存の webhook を編集するときに、webhook のトラブルシューティングを行うことができます。

{{< tabpane text=true >}}
{{% tab header="Weights & Biases App UI" value="app" %}}

Team 管理者は、Weights & Biases App UI を使用して webhook をインタラクティブにテストできます。

1. Weights & Biases Team の Settings ページに移動します。
2. **Webhooks** セクションまでスクロールします。
3. webhook の名前の横にある水平方向の 3 つのドキュメント（ミートボールアイコン）をクリックします。
4. **Test** を選択します。
5. 表示される UI パネルから、表示されるフィールドに POST リクエストを貼り付けます。
    {{< img src="/images/models/webhook_ui.png" alt="Webhook ペイロードのテストのデモ" >}}
6. **Test webhook** をクリックします。Weights & Biases App UI 内で、Weights & Biases はエンドポイントからの応答を投稿します。
    {{< img src="/images/models/webhook_ui_testing.gif" alt="Webhook のテストのデモ" >}}

デモンストレーションについては、ビデオ [Weights & Biases での Webhook のテスト](https://www.youtube.com/watch?v=bl44fDpMGJw&ab_channel=Weights%26Biases) をご覧ください。
{{% /tab %}}

{{% tab header="Bash スクリプト" value="bash"%}}

このシェルスクリプトは、トリガーされたときに Weights & Biases が webhook オートメーションに送信するリクエストと同様の `POST` リクエストを生成する 1 つの方法を示しています。

以下のコードをコピーしてシェルスクリプトに貼り付け、webhook のトラブルシューティングを行います。次の独自の値（バリュー）を指定します。

* `ACCESS_TOKEN`
* `SECRET`
* `PAYLOAD`
* `API_ENDPOINT`

{{< prism file="/webhook_test.sh" title="webhook_test.sh">}}{{< /prism >}}

{{% /tab %}}
{{< /tabpane >}}
