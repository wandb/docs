---
title: Artifact automations
description: プロジェクトに範囲を限定したアーティファクトの自動化を利用して、エイリアスやバージョンが作成または変更されたときにアクションをトリガーします。
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# アーティファクトの変更でCI/CDイベントをトリガーする

アーティファクトが変更されたときにトリガーされるオートメーションを作成します。アーティファクトのバージョン管理のために下流アクションを自動化したい場合は、アーティファクトオートメーションを使用します。オートメーションを作成するには、[イベントタイプ](#event-types)に基づいて実行したい[action](#action-types)を定義します。

アーティファクトの変更からトリガーされる一般的なユースケースには以下が含まれます：

* 評価/ホールドアウトデータセットの新しいバージョンがアップロードされた時、モデルレジストリ内の最良のトレーニングモデルを使用して推論を実行し、パフォーマンス情報を含むレポートを作成する[ローンチジョブをトリガーする](#create-a-launch-automation)。
* トレーニングデータセットの新しいバージョンが「プロダクション」としてラベル付けされた時、現在の最良のパフォーマンスを示すモデルの設定を使用して[リトレーニングローンチ](#create-a-launch-automation)ジョブをトリガーする。

:::info
アーティファクトオートメーションはプロジェクトにスコープされます。これは、プロジェクト内のイベントのみがアーティファクトオートメーションをトリガーすることを意味します。

これはW&Bモデルレジストリで作成されたオートメーションとは対照的です。モデルレジストリで作成されたオートメーションはモデルレジストリのスコープにあり、[モデルレジストリ](../model_registry/intro.md)にリンクされたモデルバージョンに対してイベントが実行されたときにトリガーされます。モデルバージョンのオートメーションの作成方法については、[モデルCI/CDのオートメーション](../model_registry/model-registry-automations.md)ページを参照してください。
:::

## イベントタイプ
*イベント* は、W&Bエコシステム内で発生する変更を指します。プロジェクト内のアーティファクトコレクションに対して2つの異なるイベントタイプを定義できます: **コレクション内でアーティファクトの新しいバージョンが作成される** と **アーティファクトエイリアスが追加される**。

:::tip
**コレクション内でアーティファクトの新しいバージョンが作成される** イベントタイプは、各アーティファクトのバージョンに対して繰り返しアクションを適用するために使用します。例えば、新しいデータセットアーティファクトバージョンが作成されたときにトレーニングジョブを自動的に開始するオートメーションを作成できます。

**アーティファクトエイリアスが追加される** イベントタイプは、特定のエイリアスがアーティファクトバージョンに適用されたときにアクティブになるオートメーションを作成するために使用します。例えば、"test-set-quality-check" エイリアスがアーティファクトに追加されたとき、そのデータセットに対して下流にプロセッシングをトリガーするアクションをトリガーするオートメーションを作成できます。
:::

## アクションタイプ
アクションはトリガーの結果として発生する応答的な変異です（内部または外部）。プロジェクト内のアーティファクトコレクションでイベントに応じて作成できるアクションには、Webhooksと[W&B Launch Jobs](../launch/intro.md)の2つのタイプがあります。

* Webhooks: HTTPリクエストを使用して外部Webサーバーと通信します。
* W&B Launch job: [Jobs](../launch/create-launch-job.md)は再利用可能で構成可能なrunテンプレートで、ローカルのデスクトップまたはKubernetes on EKS、Amazon SageMakerなどの外部コンピューティングリソースで新しい[Runs](../runs/intro.md)を迅速にローンチすることができます。

以下のセクションでは、WebhooksとW&B Launchを使用してオートメーションを作成する方法について説明します。

## Webhookオートメーションを作成する
W&B App UIを使用してアクションに基づいたWebhookを自動化します。これを行うには、まずWebhookを確立し、次にWebhookオートメーションを設定します。

:::info
アドレスレコード（Aレコード）を持つエンドポイントをWebhookに指定します。W&Bは、`[0-255].[0-255].[0-255].[0.255]`のようなIPアドレスや `localhost`として公開されるエンドポイントに接続することをサポートしていません。この制限はサーバーサイドリクエスト偽装（SSRF）攻撃およびその他関連する脅威ベクターから保護するのに役立ちます。
:::

### 認証または認可のためのシークレットを追加する
シークレットはチームレベルの変数で、資格情報、APIキー、パスワード、トークンなどの機密文字列を曖昧にするのに役立ちます。W&Bは、平文のコンテンツを保護したい任意の文字列を保存するためにシークレットを使用することを推奨します。

Webhookでシークレットを使用するには、そのシークレットをまずチームのシークレットマネージャに追加する必要があります。

:::info
* W&Bの管理者のみがシークレットを作成、編集、または削除できます。
* 外部サーバーがシークレットを使用しない場合、HTTP POSTリクエストを送信する場合はこのセクションをスキップしてください。
* シークレットは、Azure、GCP、またはAWSでの[W&Bサーバー](../hosting/intro.md)を使用する場合にも利用できます。異なるデプロイメントタイプを使用している場合は、W&Bアカウントチームに連絡して、W&Bでシークレットを使用する方法について相談してください。
:::

Webhookオートメーションを使用する際にW&Bが作成を推奨するシークレットには2つのタイプがあります：

* **アクセス トークン**: 送信者を認証してWebhookリクエストのセキュリティを強化します。
* **シークレット**: ペイロードから送信されるデータの真正性と整合性を保証します。

以下の手順に従ってWebhookを作成します：

1. W&B App UIに移動します。
2. **Team Settings**をクリックします。
3. ページをスクロールダウンして**Team secrets**セクションを見つけます。
4. **New secret**ボタンをクリックします。
5. モーダルが表示されます。**Secret name**フィールドにシークレットの名前を入力します。
6. **Secret**フィールドにシークレットを入力します。
7. （オプション）Webhookに追加のシークレットキーやトークンが必要な場合、表記5と6の手順を繰り返して別のシークレットを作成します。

Webhookを設定する際に使用するシークレットを指定します。詳細については、[Webhookの設定](#configure-a-webhook)セクションを参照してください。

:::tip
シークレットを作成したら、W&Bのワークフローで `$` でそのシークレットにアクセスできます。
:::

### Webhookの設定
Webhookを使用する前に、まずW&B App UIでそのWebhookを設定する必要があります。

:::info
* W&Bの管理者のみがW&BチームのWebhookを設定できます。
* Webhookが追加のシークレットキーやトークンを必要とする場合、事前に[1つ以上のシークレットを作成](#add-a-secret-for-authentication-or-authorization)していることを確認してください。
:::

1. W&B App UIに移動します。
2. **Team Settings**をクリックします。
3. ページをスクロールダウンして**Webhooks**セクションを見つけます。
4. **New webhook**ボタンをクリックします。
5. **Name**フィールドにWebhookの名前を入力します。
6. **URL**フィールドにWebhookのエンドポイントURLを入力します。
7. （オプション）**Secret**ドロップダウンメニューからWebhookペイロードを認証するために使用するシークレットを選択します。
8. （オプション）**Access token**ドロップダウンメニューから送信者を認証するために使用するアクセス トークンを選択します。

:::note
POSTリクエストでシークレットとアクセス トークンが指定される場所については、[Webhookのトラブルシューティング](#troubleshoot-your-webhook)セクションを参照してください。
:::

### Webhookを追加
Webhookが設定され、シークレット（オプション）がある場合、プロジェクトワークスペースに移動します。左サイドバーの **Automations** タブをクリックします。

1. **Event type** ドロップダウンから[イベントタイプ](#event-types)を選択します。
![](/images/artifacts/artifact_webhook_select_event.png)
2. **コレクション内でアーティファクトの新しいバージョンが作成される** イベントを選択した場合、**Artifact collection** ドロップダウンからオートメーションが反応するアーティファクトコレクションの名前を入力します。
![](/images/artifacts/webhook_new_version_artifact.png)
3. **Action type** ドロップダウンから **Webhooks** を選択します。
4. **Next step** ボタンをクリックします。
5. **Webhook** ドロップダウンからWebhookを選択します。
![](/images/artifacts/artifacts_webhooks_select_from_dropdown.png)
6. （オプション）JSON式エディタにペイロードを入力します。一般的なユースケースの例については、[ペイロードの例](#example-payloads)セクションを参照してください。
7. **Next step** ボタンをクリックします。
8. **Automation name** フィールドにWebhookオートメーションの名前を入力します。
![](/images/artifacts/artifacts_webhook_name_automation.png)
9. （オプション）Webhookの説明を入力します。
10. **Create automation** ボタンをクリックします。

### ペイロードの例

以下のタブには、一般的なユースケースに基づいたペイロードの例が示されています。例では、ペイロードのパラメータ内の条件オブジェクトを参照するために以下のキーが使われています：
* `${event_type}`：アクションをトリガーしたイベントタイプを指します。
* `${event_author}`：アクションをトリガーしたユーザーを指します。
* `${artifact_version}`：アクションをトリガーした特定のアーティファクトバージョンを指します。アーティファクトインスタンスとして渡されます。
* `${artifact_version_string}`：アクションをトリガーした特定のアーティファクトバージョンを指します。文字列として渡されます。
* `${artifact_collection_name}`：アーティファクトバージョンがリンクされているアーティファクトコレクションの名前を指します。
* `${project_name}`：アクションをトリガーした変更を所有するプロジェクトの名前を指します。
* `${entity_name}`：アクションをトリガーした変更を所有するエンティティの名前を指します。

<Tabs
  defaultValue="github"
  values={[
    {label: 'GitHub repository dispatch', value: 'github'},
    {label: 'Microsoft Teams notification', value: 'microsoft'},
    {label: 'Slack notifications', value: 'slack'},
  ]}>
  <TabItem value="github">

:::info
アクセス トークンにGHAワークフローをトリガーするための権限が必要です。詳細については、[GitHub Docs](https://docs.github.com/en/rest/repos/repos?#create-a-repository-dispatch-event)を参照してください。
:::

  W&BからGitHubアクションをトリガーするためにリポジトリディスパッチを送信します。例えば、`on` キーのトリガーとしてリポジトリディスパッチを受け入れるワークフローがあるとします：

  ```yaml
  on:
    repository_dispatch:
      types: BUILD_AND_DEPLOY
  ```

  リポジトリのペイロードは次のようになります：

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

:::note
Webhookペイロードの`event_type`キーは、GitHubワークフローのYAMLファイルの`types`フィールドと一致する必要があります。
:::

  テンプレート文字列の内容と配置は、オートメーションが設定されたイベントまたはモデルバージョンに依存します。`${event_type}`は"LINK_ARTIFACT" または "ADD_ARTIFACT_ALIAS" としてレンダリングされます。以下にマッピングの例を示します：

  ```json
  ${event_type} --> "LINK_ARTIFACT" or "ADD_ARTIFACT_ALIAS"
  ${event_author} --> "<wandb-user>"
  ${artifact_version} --> "wandb-artifact://_id/QXJ0aWZhY3Q6NTE3ODg5ODg3"
  ${artifact_version_string} --> "<entity>/<project_name>/<artifact_name>:<alias>"
  ${artifact_collection_name} --> "<artifact_collection_name>"
  ${project_name} --> "<project_name>"
  ${entity_name} --> "<entity>"
  ```

  テンプレート文字列を使用してW&BからGitHub Actionsや他のツールへ動的にコンテキストを渡します。これらのツールがPythonスクリプトを呼び出すことができる場合、[W&B API](../artifacts/download-and-use-an-artifact.md)を通じてW&Bアーティファクトを使用できます。

  リポジトリディスパッチについての詳細は、[GitHub Marketplaceの公式ドキュメント](https://github.com/marketplace/actions/repository-dispatch)を参照してください。

  </TabItem>
  <TabItem value="microsoft">

  Teamsチャネル用のWebhook URLを取得するために'Incoming Webhook'を設定します。以下はペイロードの例です：

  ```json
  {
  "@type": "MessageCard",
  "@context": "http://schema.org/extensions",
  "summary": "New Notification",
  "sections": [
    {
      "activityTitle": "WANDBからの通知",
      "text": "これはTeams webhookを通じて送信された例です。",
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

  ペイロードを実行時にW&Bデータを挿入するためにテンプレート文字列を使用できます（上記のTeamsの例のように）。

  </TabItem>
  <TabItem value="slack">

  Slackアプリをセットアップして、[Slack APIドキュメント](https://api.slack.com/messaging/webhooks)で説明されている手順に従ってWebhookインテグレーションを追加します。 `Bot User OAuth Token`にW&B webhookのアクセス トークンを指定してください。

  以下はペイロードの例です：

  ```json
  {
    "text": "WANDBからの新しいアラート！",
    "blocks": [
      {
        "type": "section",
        "text": {
          "type": "mrkdwn",
          "text": "アーティファクトイベント: ${event_type}"
        }
      },
      {
        "type":"section",
        "text": {
          "type": "mrkdwn",
          "text": "新しいバージョン: ${artifact_version_string}"
        }
      },
      {
        "type": "divider"
      },
      {
        "type": "section",
        "text": {
          "type": "mrkdwn",
          "text": "作成者: ${event_author}"
        }
      }
    ]
  }
  ```

  </TabItem>
</Tabs>

### Webhookのトラブルシューティング

W&B App UIまたはBashスクリプトを使用して、Webhookを対話型にトラブルシューティングできます。新しいWebhookの作成時または既存のWebhookの編集時にWebhookをトラブルシューティングできます。

<Tabs
  defaultValue="app"
  values={[
    {label: 'W&B App UI', value: 'app'},
    {label: 'Bash script', value: 'bash'},
  ]}>
  <TabItem value="app">

W&B App UIを使用してWebhookを対話的にテストします。

1. W&B Team Settingsページに移動します。
2. **Webhooks**セクションまでスクロールします。
3. Webhookの名前の横の水平の3つのドット（ミートボールアイコン）をクリックします。
4. **Test**を選択します。
5. 表示されるUIパネルから、POSTリクエストをフィールドに貼り付けます。
![](/images/models/webhook_ui.png)
6. **Test webhook**をクリックします。

W&B App UI内では、エンドポイントからのレスポンスが投稿されます。

![](/images/models/webhook_ui_testing.gif)

実際の例を見るために、[Weights & BiasesのWebhooksテスト](https://www.youtube.com/watch?v=bl44fDpMGJw&ab_channel=Weights%26Biases)のYouTube動画をご覧ください。

  </TabItem>
  <TabItem value="bash">

以下のbashスクリプトは、Webhookオートメーションがトリガーされた際にW&BがWebhookに送信するPOSTリクエストに似たリクエストを生成します。

コードを以下のようにシェルスクリプトにコピーして貼り付け、Webhookをトラブルシューティングします。以下の値を独自のものに指定します：

* `ACCESS_TOKEN`
* `SECRET`
* `PAYLOAD`
* `API_ENDPOINT`

```sh title="webhook_test.sh"
#!/bin/bash

# Your access token and secret
ACCESS_TOKEN="your_api_key" 
SECRET="your_api_secret"

# The data you want to send (for example, in JSON format)
PAYLOAD='{"key1": "value1", "key2": "value2"}'

# Generate the HMAC signature
# For security, Wandb includes the X-Wandb-Signature in the header computed 
# from the payload and the shared secret key associated with the webhook 
# using the HMAC with SHA-256 algorithm.
SIGNATURE=$(echo -n "$PAYLOAD" | openssl dgst -sha256 -hmac "$SECRET" -binary | base64)

# Make the cURL request
curl -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "X-Wandb-Signature: $SIGNATURE" \
  -d "$PAYLOAD" API_ENDPOINT
```

  </TabItem>
</Tabs>

## Launchオートメーションを作成する
W&Bジョブを自動的に開始します。

:::info
このセクションでは、すでにジョブ、キューを作成し、アクティブなエージェントがポーリングしていることを前提としています。詳細については、[W&B Launch ドキュメント](../launch/intro.md)を参照してください。
:::

1. **Event type** ドロップダウンからイベントタイプを選択します。対応するイベントについては[イベントタイプ](#event-types)セクションを参照してください。
2. （オプション）**Collection** イベントが選択されている場合は、**Artifact collection** ドロップダウンからアーティファクトコレクションの名前を入力します。
3. **Jobs** を **Action type** ドロップダウンから選択します。
4. **Next step**をクリックします。
5. **Job** ドロップダウンからW&B Launchジョブを選択します。
6. **Job version** ドロップダウンからバージョンを選択します。
7. （オプション）新しいジョブ用のハイパーパラメーターオーバーライドを入力します。
8. **Destination project** ドロップダウンからプロジェクトを選択します。
9. キューにジョブをエンキューします。
10. **Next step**をクリックします。
11. **Automation name** フィールドにWebhookオートメーションの名前を入力します。
12. （オプション）Webhookの説明を入力します。
13. **Create automation** ボタンをクリックします。

## オートメーションを表示する
W&B App UIからアーティファクトに関連するオートメーションを表示します。

1. W&B Appでプロジェクトワークスペースに移動します。
2. 左サイドバーの **Automations** タブをクリックします。

![](/images/artifacts/automations_sidebar.gif)

オートメーションセクション内では、プロジェクト内で作成された各オートメーションに対して以下のプロパティを確認できます：

- **Trigger type**：トリガーが設定されたタイプ。
- **Action type**：オートメーションをトリガーするアクションタイプ。使用できるオプションはWebhooksとLaunchです。
- **Action name**：オートメーション作成時に提供されたアクションの名前。
- **Queue**：ジョブがエンキューされたキューの名前。Webhookアクションタイプを選択した場合、このフィールドは空のままです。

## オートメーションを削除する
アーティファクトに関連するオートメーションを削除します。アクションが完了する前にオートメーションを削除した場合、進行中のアクションには影響しません。

