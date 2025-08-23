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

このページでは、Webhookオートメーションの作成方法を説明します。Slackオートメーションを作成したい場合は、[Slack オートメーションの作成]({{< relref path="/guides/core/automations/create-automations/slack.md" lang="ja" >}}) を参照してください。

Webhook オートメーションを作成する大まかな流れは以下の通りです。
1. 必要に応じて、オートメーションに必要なアクセストークン・パスワード・SSHキーなどの機密文字列ごとに、[W&B シークレットを作成]({{< relref path="/guides/core/secrets.md" lang="ja" >}})します。シークレットは**Team Settings**で定義します。
1. Webhook のエンドポイントと認証情報を定義し、インテグレーションが必要なシークレットにアクセスできるように[Webhookを作成]({{< relref path="#create-a-webhook" lang="ja" >}})します。
1. 監視する[event]({{< relref path="/guides/core/automations/automation-events.md" lang="ja" >}})や、W&Bが送信するペイロードを定義する[オートメーションを作成]({{< relref path="#create-an-automation" lang="ja" >}})します。ペイロードで必要なシークレットにはオートメーションからアクセス権を与えてください。

## Webhookの作成
チーム管理者はチーム用のWebhookを追加できます。

{{% alert %}}
Webhook で Bearer トークンや、ペイロードに機密文字列が必要な場合、事前に[それを含むシークレットを作成]({{< relref path="/guides/core/secrets.md#add-a-secret" lang="ja" >}})してください。Webhook にはアクセストークン1つとその他のシークレット1つまでしか設定できません。Webhook の認証・認可要件はサービスによって異なります。
{{% /alert %}}

1. W&B にログインし、**Team Settings**ページにアクセスします。
1. **Webhooks**セクションで、**New webhook**をクリックします。
1. Webhook の名前を入力します。
1. エンドポイントのURLを入力します。
1. Bearer トークンが必要な場合、**Access token** にその値を含む[シークレット]({{< relref path="/guides/core/secrets.md" lang="ja" >}})を指定します。Webhook オートメーション使用時、W&Bは `Authorization: Bearer` ヘッダーにアクセストークンをセットし、その値は [ペイロード変数]({{< relref path="#payload-variables" lang="ja" >}}) `${ACCESS_TOKEN}` で参照可能です。W&B が送る `POST` リクエストの構造については[Webhook のトラブルシュート]({{< relref path="#troubleshoot-your-webhook" lang="ja" >}}) を参照ください。
1. パスワードやその他の機密文字列がペイロードで必要な場合は、**Secret**にそれを含むシークレットを指定してください。Webhook を利用するオートメーションの設定時に、名前の先頭に `$` を付けて[ペイロード変数]({{< relref path="#payload-variables" lang="ja" >}})として利用できます。

    Webhook のアクセストークンがシークレットに保存されている場合は、_次のステップも_必ず実施し、アクセストークンとして指定してください。
1. W&Bがエンドポイントと通信し認証できることを確認するには:
    1. 必要に応じて、テスト用ペイロードを入力します。Webhook がアクセスできるシークレットをペイロードで参照する場合、名前の先頭に `$` を付与してください。このペイロードはテスト目的のみで保存されません。オートメーションのペイロードは、[オートメーション作成時]({{< relref path="#create-a-webhook-automation" lang="ja" >}})に設定します。`POST`リクエストのどこにシークレットやアクセストークンが含まれるかは [Webhookのトラブルシュート]({{< relref path="#troubleshoot-your-webhook" lang="ja" >}}) を参照してください。
    1. **Test** をクリックします。W&B が設定した認証情報でWebhookエンドポイントへ接続を試みます。ペイロードを入力した場合はその内容が送信されます。

    テストが失敗した場合はWebhookの設定を再確認し、必要に応じて再度お試しください。[Webhookのトラブルシュート]({{< relref path="#troubleshoot-your-webhook" lang="ja" >}}) を参照することもできます。

![Screenshot showing two webhooks in a Team](/images/automations/webhooks.png)

これで、このWebhookを利用する[オートメーションを作成]({{< relref path="#create-a-webhook-automation" lang="ja" >}})できるようになりました。

## オートメーションの作成
[Webhookを設定]({{< relref path="#create-a-webhook" lang="ja" >}})したあとは、**Registry** もしくは **Project** を選択し、Webhookをトリガーするオートメーションを作成してください。

{{< tabpane text=true >}}
{{% tab "Registry" %}}
Registry 管理者は、その Registry 内でオートメーションを作成できます。Registry オートメーションは、その Registry 内のすべてのコレクションに適用され、新しく追加されたコレクションにも適用されます。

1. W&B にログインします。
1. Registry名をクリックして詳細ページを表示します。
1. Registry全体に適用するオートメーションを作成する場合、**Automations**タブをクリックし、**Create automation**を選択します。Registryにスコープされたオートメーションは、その全コレクション（将来的に追加されるものも含む）へ自動適用されます。

    Registry内の特定のコレクションだけに適用したい場合、対象コレクションの `...` メニューから **Create automation** を選択します。また、コレクション詳細ページの**Automations**セクションにある**Create automation**ボタンから作成することも可能です。
1. 監視したい[event]({{< relref path="/guides/core/automations/automation-events.md" lang="ja" >}})を選択します。イベントによって追加入力フィールドが表示される場合があります（例: **An artifact alias is added** を選択した場合は **Alias regex** の入力が必要です）。**Next step** をクリックします。
1. [Webhook]({{< relref path="#create-a-webhook" lang="ja" >}})を所有するチームを選択します。
1. **Action type** を **Webhooks** に設定し、利用したい [Webhook]({{< relref path="#create-a-webhook" lang="ja" >}}) を選択します。
1. Webhook のアクセストークンを設定した場合は、`${ACCESS_TOKEN}` [ペイロード変数]({{< relref path="#payload-variables" lang="ja" >}}) で参照可能です。シークレットを設定した場合は、名前の先頭に `$` を付けることでペイロード内で利用できます。Webhookの要件はサービスによって異なります。
1. **Next step** をクリックします。
1. オートメーションに名前をつけ、任意で説明を追加し、**Create automation**をクリックします。

{{% /tab %}}
{{% tab "Project" %}}
W&B 管理者はプロジェクト内でオートメーションを作成できます。

1. W&B にログインし、該当プロジェクトページへ移動します。
1. サイドバーで **Automations** をクリックし、**Create automation** をクリックします。

    もしくは、ワークスペース内の折れ線グラフパネルから表示中メトリクスの[run metric オートメーション]({{< relref path="/guides/core/automations/automation-events.md#run-events" lang="ja" >}})を素早く作成できます。パネルにマウスを重ね、パネル上部のベルアイコンをクリックしてください。
    {{< img src="/images/automations/run_metric_automation_from_panel.png" alt="Automation bell icon location" >}}
1. 監視する[event]({{< relref path="/guides/core/automations/automation-events.md" lang="ja" >}})（例: アーティファクトエイリアス追加や run メトリクスが閾値に達したタイミングなど）を選択します。

    1. イベントによって追加で入力フィールドが表示される場合は、案内に従い入力します（例: **An artifact alias is added** の場合は **Alias regex** が必要です）。

    1. コレクションフィルターを省略可で指定できます。指定しない場合、オートメーションはプロジェクト内のすべてのコレクション（将来的に追加されるものも含め）に適用されます。

    **Next step** をクリックします。
1. [Webhook]({{< relref path="#create-a-webhook" lang="ja" >}})を所有するチームを選択します。
1. **Action type** を **Webhooks** に設定し、利用したい [Webhook]({{< relref path="#create-a-webhook" lang="ja" >}}) を選択します。
1. Webhook でペイロードが必要な場合は、内容を入力して **Payload** フィールドに貼り付けます。アクセストークンを設定した場合は、`${ACCESS_TOKEN}` [ペイロード変数]({{< relref path="#payload-variables" lang="ja" >}}) で参照できます。シークレットは `$<シークレット名>` として参照できます。Webhookの要件はサービスによって異なります。
1. **Next step** をクリックします。
1. オートメーションに名前をつけ、任意で説明を追加し、**Create automation**をクリックします。

{{% /tab %}}
{{< /tabpane >}}

## オートメーションの表示と管理
{{< tabpane text=true >}}
{{% tab "Registry" %}}

- Registry の **Automations** タブから、その Registry のオートメーションを管理できます。
- 各コレクションの詳細ページの **Automations** セクションから、そのコレクションのオートメーションを管理できます。

これらいずれかのページにて、Registry 管理者は既存オートメーションの管理が可能です:
- オートメーションの詳細を見るには、その名前をクリックします。
- 編集するにはアクション`...`メニューから**Edit automation**を選択します。
- 削除するにはアクション`...`メニューから**Delete automation**を選択し、確認してください。

{{% /tab %}}
{{% tab "Project" %}}
W&B 管理者は、プロジェクトの **Automations** タブから、そのプロジェクトで作成したオートメーションの表示や管理を行えます。

- オートメーションの詳細を見るには名前をクリックします。
- 編集するにはアクション`...`メニューから**Edit automation**を選択。
- 削除するにはアクション`...`メニューから**Delete automation**を選択し、確認してください。
{{% /tab %}}
{{< /tabpane >}}

## ペイロードリファレンス
ここではWebhookペイロードの構成例を示します。Webhookやそのペイロードのテスト方法は[Webhookのトラブルシュート]({{< relref path="#troubleshoot-your-webhook" lang="ja" >}}) を参照してください。

### ペイロード変数
Webhookのペイロード構築時に使える変数を紹介します。

| 変数 | 詳細 |
|----------|---------|
| `${project_name}`             | アクションをトリガーしたmutationのプロジェクト名。 |
| `${entity_name}`              | アクションをトリガーしたエンティティまたはチーム名。 |
| `${event_type}`               | アクションをトリガーしたイベントの種類。 |
| `${event_author}`             | アクションをトリガーしたユーザー。 |
| `${alias}`                    | **An artifact alias is added** イベントでトリガーされた場合、そのアーティファクトのエイリアス。それ以外では空文字。 |
| `${tag}`                      | **An artifact tag is added** イベントでトリガーされた場合、そのアーティファクトのタグ。それ以外では空文字。 |
| `${artifact_collection_name}` | アーティファクトバージョンが紐付くアーティファクトコレクション名。 |
| `${artifact_metadata.<KEY>}`  | トリガーとなったアーティファクトバージョンの任意トップレベルメタデータキーの値。`<KEY>` を、該当トップレベルメタデータキー名で置き換え。ペイロードで参照可能なのはトップレベルキーのみ。 |
| `${artifact_version}`         | アクションをトリガーしたアーティファクトバージョンの [`Wandb.Artifact`]({{< relref path="/ref/python/sdk/classes/artifact.md/" lang="ja" >}}) 表現。 |
| `${artifact_version_string}` | アクションをトリガーしたアーティファクトバージョンの `string` 表現。 |
| `${ACCESS_TOKEN}` | [Webhook]({{< relref path="#create-a-webhook" lang="ja" >}})で設定したアクセストークンの値。設定時は`Authorization: Bearer` ヘッダーでも自動渡しされます。 |
| `${SECRET_NAME}` | [Webhook]({{< relref path="#create-a-webhook" lang="ja" >}})で設定した任意のシークレット値。`SECRET_NAME` は登録したシークレット名で置換。 |

### ペイロード例
よくあるユースケースのWebhookペイロード例を示します。[ペイロード変数]({{< relref path="#payload-variables" lang="ja" >}}) の活用方法もご覧いただけます。

{{< tabpane text=true >}}
{{% tab header="GitHub repository dispatch" value="github" %}}

{{% alert %}}
アクセストークンにGHAワークフローの実行に必要な権限が付与されているかご確認ください。詳細は [GitHub ドキュメント](https://docs.github.com/en/rest/repos/repos?#create-a-repository-dispatch-event) を参照。
{{% /alert %}}

W&Bからリポジトリディスパッチを送信しGitHub アクションをトリガーします。たとえば、GitHub ワークフローファイルで `on` キーのトリガーとしてリポジトリディスパッチを受け付ける場合:

```yaml
on:
repository_dispatch:
  types: BUILD_AND_DEPLOY
```

ペイロードは次のような例になります。

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
webhookペイロードの`event_type`キーの値は、GitHubワークフローYAMLの`types`フィールドと一致する必要があります。
{{% /alert %}}

テンプレート変数の値や位置は、オートメーションの紐付け先イベントやモデルバージョンによって異なります。`${event_type}`は `LINK_ARTIFACT` または `ADD_ARTIFACT_ALIAS` が出力されます。以下は値マッピングの例です。

```text
${event_type} --> "LINK_ARTIFACT" または "ADD_ARTIFACT_ALIAS"
${event_author} --> "<wandb-user>"
${artifact_version} --> "wandb-artifact://_id/QXJ0aWZhY3Q6NTE3ODg5ODg3""
${artifact_version_string} --> "<entity>/model-registry/<registered_model_name>:<alias>"
${artifact_collection_name} --> "<registered_model_name>"
${project_name} --> "model-registry"
${entity_name} --> "<entity>"
```

テンプレート文字列を使えばW&BからGitHub Actionsその他のツールへ情報を動的に渡せます。これらツールが Python スクリプトを実行できる場合、[W&B API]({{< relref path="/guides/core/artifacts/download-and-use-an-artifact.md" lang="ja" >}})を通じて登録済みモデルアーティファクトも取得可能です。

- Repository dispatchについて詳しくは[GitHub Marketplace公式ドキュメント](https://github.com/marketplace/actions/repository-dispatch)をご覧ください。

- モデル評価のためのWebhookオートメーション: [Webhook Automations for Model Evaluation](https://www.youtube.com/watch?v=7j-Mtbo-E74&ab_channel=Weights%26Biases)
- モデルデプロイメントのためのWebhookオートメーション: [Webhook Automations for Model Deployment](https://www.youtube.com/watch?v=g5UiAFjM2nA&ab_channel=Weights%26Biases)
  でガイド動画もご覧いただけます。

- W&Bによる[レポート](https://wandb.ai/wandb/wandb-model-cicd/reports/Model-CI-CD-with-W-B--Vmlldzo0OTcwNDQw)では、Model CI 用GitHub Actions Webhook活用例が紹介されています。[本リポジトリ](https://github.com/hamelsmu/wandb-modal-webhook)でも Modal Labs Webhook で Model CI を構成する方法が学べます。

{{% /tab %}}

{{% tab header="Microsoft Teams notification" value="microsoft"%}}

以下は Teams のWebhookでチャンネルに通知するペイロード例です。

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

このようにテンプレート変数を使うことで、W&Bデータを実行時にペイロード内へ挿入できます（上記Teams例参照）。

{{% /tab %}}

{{% tab header="Slack notifications" value="slack"%}}

{{% alert %}}
このセクションは過去の事例として掲載しています。WebhookでSlack連携をお使いの場合は、[新しいSlackインテグレーション]({{ relref "#create-a-slack-automation"}}) の利用を推奨します。
{{% /alert %}}

SlackアプリのセットアップおよびWebhook連携の追加については、[Slack APIドキュメント](https://api.slack.com/messaging/webhooks)を参考に進めてください。`Bot User OAuth Token` に登録された値をW&B Webhookのアクセストークンとして指定しているかご確認ください。

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

## Webhook のトラブルシュート
Webhook は W&B App UI からインタラクティブに、または Bash スクリプトでプログラム的にトラブルシュートできます。Webhook の新規作成時および編集時のいずれでも対応可能です。

W&B が `POST` リクエストで使うフォーマットの詳細は **Bash script** タブをご覧ください。

{{< tabpane text=true >}}
{{% tab header="W&B App UI" value="app" %}}

チーム管理者は W&B App UI からWebhook のテストが可能です。

1. W&B の Team Settings ページにアクセスします。
2. **Webhooks** セクションまでスクロールします。
3. Webhook名の横にある3点リーダ（水玉アイコン）をクリックします。
4. **Test** を選択します。
5. 表示されたパネルの入力欄に POST リクエスト内容を貼り付けます。
    {{< img src="/images/models/webhook_ui.png" alt="Demo of testing a webhook payload" >}}
6. **Test webhook** をクリックすると、エンドポイントのレスポンスがW&B App UIに表示されます。
    {{< img src="/images/models/webhook_ui_testing.gif" alt="Demo of testing a webhook" >}}

実際の操作デモは[Testing Webhooks in W&B](https://www.youtube.com/watch?v=bl44fDpMGJw&ab_channel=Weights%26Biases)の動画もご参照ください。
{{% /tab %}}

{{% tab header="Bash script" value="bash"%}}

以下のシェルスクリプトは、W&BがWebhookオートメーションをトリガーする際に送信する`POST`リクエストの一例を示しています。

次の値を、ご自身の環境に合わせて指定してください。

* `ACCESS_TOKEN`
* `SECRET`
* `PAYLOAD`
* `API_ENDPOINT`

{{< prism file="/webhook_test.sh" title="webhook_test.sh">}}{{< /prism >}}

{{% /tab %}}
{{< /tabpane >}}