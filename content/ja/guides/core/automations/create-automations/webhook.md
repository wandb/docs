---
title: Webhook オートメーションを作成する
menu:
  default:
    identifier: create-webhook-automations
    parent: automations
weight: 3
---

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

このページでは、Webhook オートメーションの作成方法を説明します。Slack オートメーションを作成したい場合は、[Slack オートメーションの作成]({{< relref "/guides/core/automations/create-automations/slack.md" >}})をご覧ください。

Webhook オートメーション作成の全体的な流れは次のとおりです。
1. 必要に応じて、オートメーションで必要なアクセストークンやパスワード、SSH キーなどの機密文字列ごとに [W&B シークレットを作成]({{< relref "/guides/core/secrets.md" >}})します。シークレットは **Team Settings** で定義します。
1. [Webhook を作成]({{< relref "#create-a-webhook" >}})し、エンドポイントや認証の詳細を設定し、インテグレーションが必要なシークレットへアクセスできるようにします。
1. [オートメーションを作成]({{< relref "#create-an-automation" >}})し、監視対象の [イベント]({{< relref "/guides/core/automations/automation-events.md" >}})や W&B から送信されるペイロードを定義します。必要に応じて、オートメーションにもシークレットを渡します。

## Webhook の作成

チーム管理者はチームの Webhook を追加できます。

{{% alert %}}
Webhook が Bearer トークンやペイロード内に機微な文字列を要する場合は、[それを含むシークレットを作成]({{< relref "/guides/core/secrets.md#add-a-secret" >}})してから Webhook を作成してください。Webhook にはアクセストークン 1 件と他のシークレット 1 件まで設定できます。Webhook の認証・権限要件は、そのサービスに依存します。
{{% /alert %}}

1. W&B にログインし、**Team Settings** ページへ移動します。
1. **Webhooks** セクションで **New webhook** をクリックします。
1. Webhook の名前を入力します。
1. Webhook のエンドポイント URL を入力します。
1. Webhook が Bearer トークンを要求する場合、**Access token** にその値を含む [シークレット]({{< relref "/guides/core/secrets.md" >}})を設定します。Webhook オートメーション利用時、W&B が `Authorization: Bearer` HTTP ヘッダーにアクセストークンを自動でセットし、`${ACCESS_TOKEN}` [ペイロード変数]({{< relref "#payload-variables" >}})としても利用できます。W&B が Webhook サービスに送る `POST` リクエストの構造は、[Webhook のトラブルシューティング]({{< relref "#troubleshoot-your-webhook" >}})を参照してください。
1. ペイロード内でパスワードやその他の機密文字列が必要な場合、**Secret** に該当のシークレットを設定します。オートメーション設定時、ペイロード内で `$` を先頭につけて [ペイロード変数]({{< relref "#payload-variables" >}})のように利用できます。

    アクセストークンをシークレットに保存した場合、次の手順でそのシークレットをアクセストークンとしても指定してください。
1. W&B がエンドポイントへ接続・認証できることを確認するために：
    1. 必要に応じてテスト用のペイロードを入力します。Webhook がアクセスできるシークレットを参照したい場合、ペイロード内で `$` を先頭につけて指定します。このペイロードはテスト専用で保存はされません。オートメーション本体のペイロードは[オートメーション作成時]({{< relref "#create-a-webhook-automation" >}})に設定します。`POST` リクエストでシークレットやアクセストークンがどこで指定されているかは [Webhook のトラブルシューティング]({{< relref "#troubleshoot-your-webhook" >}}) を参照してください。
    1. **Test** をクリックします。W&B が設定した認証情報でエンドポイントに接続を試み、ペイロードがある場合はそれも送信します。

    テストがうまくいかない場合は、Webhook の設定を見直し、もう一度お試しください。必要に応じて [Webhook のトラブルシューティング]({{< relref "#troubleshoot-your-webhook" >}}) を参照してください。

![Screenshot showing two webhooks in a Team](/images/automations/webhooks.png)

Webhook を利用する [オートメーションを作成]({{< relref "#create-a-webhook-automation" >}})できるようになります。

## オートメーションの作成

[Webhook を設定した]({{< relref "#create-a-webhook" >}})ら、**Registry** または **Project** のいずれかを選択し、Webhook をトリガーするオートメーションを以下の手順で作成できます。

{{< tabpane text=true >}}
{{% tab "Registry" %}}
Registry 管理者はレジストリ内でオートメーションを作成できます。レジストリのオートメーションは、そのレジストリ下のすべてのコレクション（将来的に追加されたものも含む）に自動適用されます。

1. W&B にログインします。
1. 対象レジストリ名をクリックして詳細ページへ入ります。
1. レジストリ全体のオートメーションを作成する場合は、**Automations** タブを開き、**Create automation** をクリックします。レジストリ単位のオートメーションは、配下のすべてのコレクション（今後追加されたものも含む）に自動で適用されます。

    レジストリ内の特定コレクションだけにスコープしたオートメーションを作成する場合は、そのコレクションの `...` メニューから **Create automation** を選びます。またはコレクションの詳細ページの **Automations** セクションにある **Create automation** ボタンから作成できます。
1. 監視したい [イベント]({{< relref "/guides/core/automations/automation-events.md" >}})を選択します。イベントに応じて追加フィールドが現れます（例：**An artifact alias is added** を選んだ場合は **Alias regex** を指定）。**Next step** をクリック。
1. [Webhook を所有するチーム]({{< relref "#create-a-webhook" >}})を選択します。
1. **Action type** を **Webhooks** に設定し、利用する [Webhook]({{< relref "#create-a-webhook" >}}) を選択します。
1. アクセストークンを設定した場合、`${ACCESS_TOKEN}` [ペイロード変数]({{< relref "#payload-variables" >}})でトークンにアクセス可能です。Webhook 用のシークレットも設定していれば、ペイロード内で `$` を頭に付けてアクセスできます。Webhook 側の要件によります。
1. **Next step** をクリックします。
1. オートメーションの名前を入力し、必要に応じて説明も追加します。**Create automation** をクリックします。

{{% /tab %}}
{{% tab "Project" %}}
W&B 管理者はプロジェクト内でオートメーションを作成できます。

1. W&B にログインし、プロジェクトページへ進みます。
1. サイドバーから **Automations** を開き、**Create automation** をクリックします。

    またはワークスペース内のラインプロットから、その指標に対して [run metric オートメーション]({{< relref "/guides/core/automations/automation-events.md#run-events" >}})を素早く作成できます。パネルにカーソルを合わせ、パネル上部のベルアイコンをクリックします。
    {{< img src="/images/automations/run_metric_automation_from_panel.png" alt="Automation bell icon location" >}}
1. 監視対象の [イベント]({{< relref "/guides/core/automations/automation-events.md" >}}) を選択します（例：アーティファクトエイリアスが追加された時、run 指標がしきい値を超えた時など）。

    1. イベントに応じて追加フィールドを入力します（例：**An artifact alias is added** を選ぶ場合は **Alias regex** を指定）。

    1. 必要に応じてコレクションフィルタを指定します。指定しない場合、オートメーションはプロジェクト内の全コレクション（将来追加分含む）に適用されます。

    **Next step** をクリックします。
1. [Webhook を所有するチーム]({{< relref "#create-a-webhook" >}})を選択します。
1. **Action type** を **Webhooks** に設定し、利用する [Webhook]({{< relref "#create-a-webhook" >}}) を選びます。
1. Webhook にペイロードが必要な場合は、構築して **Payload** フィールドに貼り付けます。アクセストークンやシークレットを設定した場合、それぞれ `${ACCESS_TOKEN}` [ペイロード変数]({{< relref "#payload-variables" >}}) または `$` プレフィックス付きで利用できます。Webhook 側の仕様をご確認ください。
1. **Next step** をクリックします。
1. オートメーション名を入力し、任意で説明も加えて、**Create automation** をクリックします。

{{% /tab %}}
{{< /tabpane >}}

## オートメーションの表示と管理

{{< tabpane text=true >}}
{{% tab "Registry" %}}

- レジストリ管理者は、レジストリの **Automations** タブからオートメーションを管理できます。
- 各コレクションの詳細ページの **Automations** セクションからも管理可能です。

いずれのページからも既存オートメーションの管理ができます。
- 詳細を確認したい場合はオートメーション名をクリック。
- 編集する場合は `...` メニューから **Edit automation** を選択。
- 削除する場合も `...` メニューから **Delete automation** を選択（確認が求められます）。

{{% /tab %}}
{{% tab "Project" %}}
W&B 管理者はプロジェクトの **Automations** タブからオートメーションの表示と管理が可能です。

- 詳細を見るにはオートメーション名をクリック。
- 編集は `...` メニューから **Edit automation** を選択。
- 削除も `...` メニューから **Delete automation** を選択（確認が必要）。
{{% /tab %}}
{{< /tabpane >}}

## ペイロードリファレンス

このセクションを使って Webhook のペイロードを構築できます。Webhook やペイロードのテスト方法は [Webhook のトラブルシューティング]({{< relref "#troubleshoot-your-webhook" >}}) をご覧ください。

### ペイロード変数

このセクションでは Webhook のペイロード構築時に利用できる変数を説明します。

| 変数 | 詳細 |
|----------|---------|
| `${project_name}`             | アクションをトリガーした変更元プロジェクトの名前 |
| `${entity_name}`              | アクションをトリガーした変更元 Entity またはチームの名前 |
| `${event_type}`               | アクションをトリガーしたイベントタイプ |
| `${event_author}`             | アクションをトリガーしたユーザー |
| `${alias}`                    | **An artifact alias is added** イベントでトリガーされた場合はエイリアスが入ります。他のオートメーションでは空です。 |
| `${tag}`                      | **An artifact tag is added** イベントでトリガーされた場合はタグが入ります。他のオートメーションでは空です。 |
| `${artifact_collection_name}` | 対象アーティファクトバージョンが紐づくアーティファクトコレクション名 |
| `${artifact_metadata.<KEY>}`  | トリガーとなったアーティファクトバージョンのトップレベルメタデータ キーの値。`<KEY>` をキー名で置き換えて下さい。Webhook のペイロードにはトップレベルのメタデータキーのみ利用可。|
| `${artifact_version}`         | トリガーとなったアーティファクトバージョンの [`Wandb.Artifact`]({{< relref "/ref/python/sdk/classes/artifact.md/" >}}) 表現 |
| `${artifact_version_string}` | トリガーとなったアーティファクトバージョンの `string` 表現 |
| `${ACCESS_TOKEN}` | [Webhook]({{< relref "#create-a-webhook" >}}) で設定されたアクセストークンがある場合はその値。`Authorization: Bearer` HTTP ヘッダでも自動的に渡されます。|
| `${SECRET_NAME}` | 設定した場合は [Webhook]({{< relref "#create-a-webhook" >}}) で指定したシークレット値。`SECRET_NAME` をシークレット名に置き換えて使用します。|

### ペイロード例

このセクションでは代表的なユースケースにおける Webhook ペイロードの例を紹介します。サンプルは [ペイロード変数]({{< relref "#payload-variables" >}}) の使い方を示しています。

{{< tabpane text=true >}}
{{% tab header="GitHub repository dispatch" value="github" %}}

{{% alert %}}
GHA ワークフローをトリガーするには、アクセストークンに必要な権限があるかご確認ください。詳細は [GitHub Docs](https://docs.github.com/en/rest/repos/repos?#create-a-repository-dispatch-event) を参照してください。
{{% /alert %}}

W&B から GitHub へ repository dispatch を送信し、GitHub Action をトリガーする例です。例えば、以下のような repository dispatch をトリガーとして受け付けるワークフローファイルがあった場合：

```yaml
on:
repository_dispatch:
  types: BUILD_AND_DEPLOY
```

repository 向けのペイロード例：

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
Webhook ペイロードの `event_type` キーは、GitHub ワークフロー YAML ファイル内の `types` フィールドと一致させてください。
{{% /alert %}}

テンプレート文字列の内容や位置は、設定するイベントやモデルバージョンによって異なります。`${event_type}` は `LINK_ARTIFACT` または `ADD_ARTIFACT_ALIAS` として展開されます。例：

```text
${event_type} --> "LINK_ARTIFACT" または "ADD_ARTIFACT_ALIAS"
${event_author} --> "<wandb-user>"
${artifact_version} --> "wandb-artifact://_id/QXJ0aWZhY3Q6NTE3ODg5ODg3""
${artifact_version_string} --> "<entity>/model-registry/<registered_model_name>:<alias>"
${artifact_collection_name} --> "<registered_model_name>"
${project_name} --> "model-registry"
${entity_name} --> "<entity>"
```

テンプレート文字列で W&B から GitHub Actions や他ツールへ動的に情報を渡せます。渡された情報で Python スクリプトを呼び出すことも可能なため、[W&B API]({{< relref "/guides/core/artifacts/download-and-use-an-artifact.md" >}}) で Registered Model Artifacts を利用できます。

- repository dispatch については [GitHub Marketplace の公式ドキュメント](https://github.com/marketplace/actions/repository-dispatch) をご参照ください。

- モデル評価向け [Webhook Automations for Model Evaluation](https://www.youtube.com/watch?v=7j-Mtbo-E74&ab_channel=Weights%26Biases) や モデルデプロイ向け [Webhook Automations for Model Deployment](https://www.youtube.com/watch?v=g5UiAFjM2nA&ab_channel=Weights%26Biases) のビデオも参考にしてください。

- W&B [レポート](https://wandb.ai/wandb/wandb-model-cicd/reports/Model-CI-CD-with-W-B--Vmlldzo0OTcwNDQw) でも Github Actions Webhook Automation を用いた Model CI の事例を紹介しています。 [GitHub リポジトリ](https://github.com/hamelsmu/wandb-modal-webhook) では Modal Labs Webhook 連携による Model CI 例も見られます。

{{% /tab %}}

{{% tab header="Microsoft Teams notification" value="microsoft"%}}

このペイロード例は Teams チャンネルへ Webhook で通知を送るサンプルです：

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

このようにテンプレート文字列を使うことで、実行タイミングで W&B のデータをペイロードに差し込んで通知できます。

{{% /tab %}}

{{% tab header="Slack notifications" value="slack"%}}

{{% alert %}}
このセクションは参考用です。現在 Slack との連携で Webhook をご利用の場合は、[新しい Slack インテグレーション]({{ relref "#create-a-slack-automation"}}) への更新を推奨します。
{{% /alert %}}

[Slack API documentation](https://api.slack.com/messaging/webhooks) を参考に Slack アプリに incoming webhook を追加してください。`Bot User OAuth Token` に指定したシークレットを W&B Webhook のアクセストークンとして設定してください。

以下がペイロードのサンプルです：

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

## Webhook のトラブルシューティング

Webhook のトラブルシューティングは W&B App UI のインタラクティブ操作または Bash スクリプトを使ってプログラム的に行うことができます。Webhook 作成時や既存の Webhook 編集時にもテスト可能です。

W&B が送信する `POST` リクエストの書式は「**Bash script**」タブをご確認ください。

{{< tabpane text=true >}}
{{% tab header="W&B App UI" value="app" %}}

チーム管理者は W&B App UI からインタラクティブに Webhook のテストが可能です。

1. W&B の **Team Settings** ページに移動します。
2. **Webhooks** セクションまでスクロールします。
3. Webhook 名の横にある三点リーダー（ミートボールアイコン）をクリックします。
4. **Test** を選択します。
5. 表示されたパネルに、POST リクエストを貼り付けます。
    {{< img src="/images/models/webhook_ui.png" alt="Demo of testing a webhook payload" >}}
6. **Test webhook** をクリックすると、W&B App UI 上にエンドポイントの応答が表示されます。
    {{< img src="/images/models/webhook_ui_testing.gif" alt="Demo of testing a webhook" >}}

[Testing Webhooks in W&B](https://www.youtube.com/watch?v=bl44fDpMGJw&ab_channel=Weights%26Biases) のビデオも参考にしてください。
{{% /tab %}}

{{% tab header="Bash script" value="bash"%}}

このシェルスクリプトは、W&B からあなたの Webhook オートメーションへ送信される `POST` リクエストに近い形のリクエストを試す例です。

以下をコピーしてシェルスクリプトとして保存し、次の変数を自分の値で指定してください。

* `ACCESS_TOKEN`
* `SECRET`
* `PAYLOAD`
* `API_ENDPOINT`

{{< prism file="/webhook_test.sh" title="webhook_test.sh">}}{{< /prism >}}

{{% /tab %}}
{{< /tabpane >}}