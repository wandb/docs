---
title: Model registry automations
description: モデル CI（自動化されたモデルの評価パイプライン）やモデルデプロイメントにオートメーションを使用します。
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# モデルレジストリの変更でCI/CDイベントをトリガーする

自動化することで、ワークフローのステップ、例えば自動モデルテストやデプロイをトリガーすることができます。自動化を作成するには、[イベントタイプ](#event-types)に基づいて発生させたい[アクション](#action-types)を定義します。

例えば、新しいバージョンの登録済みモデルを追加した時に、自動でモデルをGitHubにデプロイするトリガーを作成することができます。

:::info
オートメーションのための参考チュートリアルをお探しですか？
1. [こちら](https://wandb.ai/wandb/wandb-model-cicd/reports/Model-CI-CD-with-W-B--Vmlldzo0OTcwNDQw)のチュートリアルは、モデルの評価とデプロイのためのGitHubアクションをトリガーするオートメーションの設定方法を示しています
2. [こちら](https://youtube.com/playlist?list=PLD80i8An1OEGECFPgY-HPCNjXgGu-qGO6&feature=shared)のビデオシリーズでは、Webhookの基本とW&Bでの設定方法を紹介しています
3. [こちら](https://www.youtube.com/watch?v=s5CMj_w3DaQ)のデモでは、Sagemaker Endpointへのモデルデプロイのオートメーションの設定方法を詳述しています
:::

## イベントタイプ
*イベント*とは、W&Bエコシステム内で発生する変更のことです。モデルレジストリは、**新しいアーティファクトを登録済みモデルにリンクする**および**登録済みモデルのバージョンに新しいエイリアスを追加する**という2つのイベントタイプをサポートしています。

モデルバージョンのリンク方法については[Link a model version](./link-model-version.md)を、アーティファクトのエイリアスに関する情報については[Create a custom alias](../artifacts/create-a-custom-alias.md)を参照してください。

:::tip
新しいモデル候補をテストするためには、**新しいアーティファクトを登録済みモデルにリンクする**イベントタイプを使用します。`deploy`のような特定のワークフローのステップを示すエイリアスを指定するには、**登録済みモデルのバージョンに新しいエイリアスを追加する**イベントタイプを使用します。
:::

## アクションタイプ
アクションは、あるトリガーの結果として発生する応答的な変化（内部または外部）です。モデルレジストリで作成できるアクションには2種類あります: [webhooks](#create-a-webhook-automation)と [W&B Launch Jobs](../launch/intro.md)です。

* Webhooks: HTTPリクエストを使用してW&Bから外部のWebサーバーと通信します。
* W&B Launch Job: [Jobs](../launch/create-launch-job.md)は、ローカルデスクトップやKubernetes on EKS、Amazon SageMakerなどの外部コンピュートリソース上で新しい[runs](../runs/intro.md)を迅速に開始するための再利用可能で設定可能なテンプレートです。

以下のセクションでは、webhooksおよびW&B Launchを使用してオートメーションを作成する方法を説明します。

## Webhookオートメーションの作成
W&B App UIを使用して、アクションに基づいてWebhookを自動化します。これを行うには、まずWebhookを確立し、その後Webhookオートメーションを設定します。

:::info
Webhookのエンドポイントには、アドレスレコード(Aレコード)を持つエンドポイントを指定します。W&Bは、`[0-255].[0-255].[0-255].[0.255]`のような直接IPアドレスで公開されているエンドポイントや`localhost`として公開されているエンドポイントへの接続をサポートしていません。この制限は、サーバーサイドリクエストフォージェリー(SSRF)攻撃やその他の関連する脅威ベクトルから保護するためです。
:::

### 認証または認可のためにシークレットを追加する
シークレットはチームレベルの変数で、認証情報、APIキー、パスワード、トークンなどのプライベートな文字列を隠蔽するためのものです。W&Bは、プレーンテキストの内容を保護するために任意の文字列を保存する際にシークレットを使用することを推奨します。

Webhookでシークレットを使用するには、まずそのシークレットをチームのシークレットマネージャーに追加する必要があります。

:::info
* シークレットを作成、編集、削除できるのはW&B管理者のみです。
* 送信先のHTTP POST リクエストがシークレットを使用しない場合は、このセクションをスキップしてください。
* Azure、GCP、またはAWSデプロイメントで[W&B Server](../hosting/intro.md)を使用している場合もシークレットが利用可能です。異なるデプロイメントタイプを使用している場合のシークレットの使用方法については、W&Bアカウントチームにお問い合わせください。
:::

W&BがWebhookオートメーションで使用することを推奨するシークレットには2種類があります:

* **アクセストークン**: 送信者を認証してWebhookリクエストを保護する
* **シークレット**: ペイロードから送信されたデータの真正性と完全性を保証する

以下の手順に従ってWebhookを作成してください:

1. W&B App UIに移動します。
2. **Team Settings**をクリックします。
3. ページをスクロールして**Team secrets**セクションを見つけます。
4. **New secret**ボタンをクリックします。
5. モーダルが表示されます。**Secret name**フィールドにシークレットの名前を入力します。
6. **Secret**フィールドにシークレットを追加します。
7. (任意) Webhookが追加のシークレットキーまたはトークンを必要とする場合は、ステップ5と6を繰り返して別のシークレット（例えばアクセストークン）を作成します。

Webhookを設定する際に使用するシークレットを指定します。詳細については、[Configure a webhook](#configure-a-webhook)セクションを参照してください。

:::tip
シークレットを作成すると、W&Bワークフロー内で`$`を使用してそのシークレットにアクセスできます。
:::

:::caution
W&B Serverでシークレットを使用する場合の考慮事項:

セキュリティ要件を満たすセキュリティ対策を構成するのはユーザーの責任です。

W&Bは、シークレットをAWS、GCP、またはAzureが提供するクラウドシークレットマネージャーのW&Bインスタンスに保存することを強く推奨します。AWS、GCP、およびAzureが提供するシークレットマネージャーは、高度なセキュリティ機能を備えています。

シークレットストアのバックエンドとしてKubernetesクラスターを使用することは推奨しません。Kubernetesクラスターの使用を検討するのは、AWS、GCP、またはAzureのクラウドシークレットマネージャーのW&Bインスタンスを使用できず、クラスターを使用する場合に発生する可能性のあるセキュリティ脆弱性を防ぐ方法を理解している場合に限ります。
:::

### Webhookの設定
Webhookを使用する前に、まずW&B App UIでそのWebhookを設定します。

:::info
* WebhookをW&Bチームのために設定できるのはW&B管理者のみです。
* Webhookが追加のシークレットキーまたはトークンを必要とする場合は、すでに[1つ以上のシークレットを作成していること](#add-a-secret-for-authentication-or-authorization)を確認してください。
:::

1. W&B App UIに移動します。
2. **Team Settings**をクリックします。
3. ページをスクロールして**Webhooks**セクションを見つけます。
4. **New webhook**ボタンをクリックします。
5. **Name**フィールドにWebhookの名前を入力します。
6. **URL**フィールドにWebhookのエンドポイントURLを入力します。
7. (任意) **Secret**ドロップダウンメニューから、Webhookペイロードを認証するために使用するシークレットを選択します。
8. (任意) **Access token**ドロップダウンメニューから、送信者を認証するために使用するアクセストークンを選択します。
9. (任意) **Access token**ドロップダウンメニューから、Webhookを認証するために必要な追加のシークレットキーまたはトークンを選択します（例えばアクセストークン）。

:::note
POSTリクエストにおけるシークレットとアクセストークンの指定場所については、[Webhookのトラブルシューティング](#troubleshoot-your-webhook)セクションを参照してください。
:::


### Webhookの追加
Webhookが設定され（任意でシークレットも設定されている）場合、Model Registry Appにアクセスします [https://wandb.ai/registry/model](https://wandb.ai/registry/model)。

1. **Event type**ドロップダウンから[イベントタイプ](#event-types)を選択します。
![](/images/models/webhook_select_event.png)
2. (任意) **A new version is added to a registered model**イベントを選択した場合、**Registered model**ドロップダウンから登録されたモデルの名前を指定します。
![](/images/models/webhook_new_version_reg_model.png)
3. **Action type**ドロップダウンから**Webhooks**を選択します。
4. **Next step**ボタンをクリックします。
5. **Webhook**ドロップダウンからWebhookを選択します。
![](/images/models/webhooks_select_from_dropdown.png)
6. (任意) JSONエクスプレッションエディタにペイロードを入力します。一般的なユースケースの例については、[Example payload](#example-payloads)セクションを参照してください。
7. **Next step**をクリックします。
8. **Automation name**フィールドにWebhookオートメーションの名前を入力します。
![](/images/models/webhook_name_automation.png)
9. (任意) Webhookの説明を入力します。
10. **Create automation**ボタンをクリックします。

### ペイロードの例

一般的なユースケースに基づいたペイロードの例を以下のタブで示します。例の中で、ペイロードパラメータの条件オブジェクトを指す次のキーが参照されています:
* `${event_type}` アクションをトリガーしたイベントの種類を指します。
* `${event_author}` アクションをトリガーしたユーザーを指します。
* `${artifact_version}` アクションをトリガーした特定のアーティファクトバージョンを指します。アーティファクトインスタンスとして渡されます。
* `${artifact_version_string}` アクションをトリガーした特定のアーティファクトバージョンを指します。文字列として渡されます。
* `${artifact_collection_name}` アーティファクトバージョンがリンクされているアーティファクトコレクションの名前を指します。
* `${project_name}` アクションをトリガーしたミューテーションを所有するプロジェクトの名前を指します。
* `${entity_name}` アクションをトリガーしたミューテーションを所有するエンティティの名前を指します。

<Tabs
  defaultValue="github"
  values={[
    {label: 'GitHub repository dispatch', value: 'github'},
    {label: 'Microsoft Teams notification', value: 'microsoft'},
    {label: 'Slack notifications', value: 'slack'},
  ]}>
  <TabItem value="github">

:::info
アクセストークンに必要な権限があることを確認して、GHAワークフローをトリガーしてください。詳細については、[こちらのGitHubドキュメント](https://docs.github.com/en/rest/repos/repos?#create-a-repository-dispatch-event)を参照してください。
:::

W&Bからリポジトリディスパッチを送信してGitHubアクションをトリガーします。例えば、`on`キーのトリガーとしてリポジトリディスパッチを受け入れるワークフローがあるとします:

  ```yaml
  on:
    repository_dispatch:
      types: BUILD_AND_DEPLOY
  ```

リポジトリのペイロードは次のようになります:

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
Webhookペイロード内の`event_type`キーは、GitHubワークフローファイルの`types`フィールドと一致する必要があります。
:::

テンプレート文字列の内容と配置は、オートメーションが設定されているイベントまたはモデルバージョンによって異なります。`${event_type}`は"LINK_ARTIFACT"または"ADD_ARTIFACT_ALIAS"としてレンダリングされます。以下は例のマッピングです:

  ```json
  ${event_type} --> "LINK_ARTIFACT"または"ADD_ARTIFACT_ALIAS"
  ${event_author} --> "<wandb-user>"
  ${artifact_version} --> "wandb-artifact://_id/QXJ0aWZhY3Q6NTE3ODg5ODg3"
  ${artifact_version_string} --> "<entity>/model-registry/<registered_model_name>:<alias>"
  ${artifact_collection_name} --> "<registered_model_name>"
  ${project_name} --> "model-registry"
  ${entity_name} --> "<entity>"
  ```

テンプレート文字列を使用して、W&BからGitHub Actionsやその他のツールへ動的にコンテキストを渡すことができます。それらのツールがPythonスクリプトを呼び出すことができる場合、[W&B API](../artifacts/download-and-use-an-artifact.md)を通じて登録モデルアーティファクトを消費することができます。

リポジトリディスパッチについて詳しくは、[GitHub Marketplaceの公式ドキュメント](https://github.com/marketplace/actions/repository-dispatch)を参照してください。

[Webhook Automations for Model Evaluation](https://www.youtube.com/watch?v=7j-Mtbo-E74&ab_channel=Weights%26Biases)と[Webhook Automations for Model Deployment](https://www.youtube.com/watch?v=g5UiAFjM2nA&ab_channel=Weights%26Biases)のYouTubeビデオを参照して、モデル評価とデプロイのためのオートメーションを作成するステップバイステップのビデオをご覧ください。

このW&B[レポート](https://wandb.ai/wandb/wandb-model-cicd/reports/Model-CI-CD-with-W-B--Vmlldzo0OTcwNDQw)を参照して、GitHub Actions Webhook Automation for Model CIの使用方法を学びます。この[GitHubリポジトリ](https://github.com/hamelsmu/wandb-modal-webhook)をチェックして、Modal Labs WebhookでModel CIを作成する方法を学びましょう。

  </TabItem>
  <TabItem value="microsoft">

Teams ChannelのWebhook URLを取得するには、'Incoming Webhook'を設定してください。以下はペイロードの例です:

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
実行時にW&Bデータをペイロードに注入するためにテンプレート文字列を使用できます（上記のTeamsの例のように）。

  </TabItem>
  <TabItem value="slack">

Slackアプリを設定し、[Slack APIドキュメント](https://api.slack.com/messaging/webhooks)に記載されている手順に従って、インカミングWebhookインテグレーションを追加してください。`Bot User OAuth Token`として指定されたシークレットがW&B Webhookのアクセストークンとして設定されていることを確認してください。

以下はペイロードの例です:

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

  </TabItem>
</Tabs>

### Webhookのトラブルシューティング

W&B App UIまたはBashスクリプトを使用してインタラクティブにWebhookをトラブルシュートします。Webhookを新しく作成する際や既存のWebhookを編集する際にトラブルシュートできます。

<Tabs
  defaultValue="app"
  values={[
    {label: 'W&B App UI', value: 'app'},
    {label: 'Bash script', value: 'bash'},
  ]}>
  <TabItem value="app">

W&B App UIを使用してインタラクティブにWebhookをテストします。

1. W&B Team Settingsページに移動します。
2. **Webhooks**セクションまでスクロールします。
3. Webhookの名前の横にある水平の3つのドット（ミートボールアイコン）をクリックします。
4. **Test**を選択します。
5. 表示されたUIパネルにPOSTリクエストをフィールドに貼り付けます。
![](/images/models/webhook_ui.png)
6. **Test webhook**をクリックします。

W&B App UI内で、W&Bはエンドポイントによって行われたレスポンスを投稿します。

![](/images/models/webhook_ui_testing.gif)

[Testing Webhooks in Weights & Biases](https://www.youtube.com/watch?v=bl44fDpMGJw&ab_channel=Weights%26Biases)のYouTubeビデオを参照して、現実世界の例を確認してください。

  </TabItem>
  <TabItem value="bash">

以下のbashスクリプトは、W&BがWebhookオートメーションをトリガーしたときに送信するPOSTリクエストを生成します。

以下のコードをシェルスクリプトにコピー＆ペーストしてWebhookをトラブルシュートします。次の値に自分の値を指定してください。

* `ACCESS_TOKEN`
* `SECRET`
* `PAYLOAD`
* `API_ENDPOINT`
```sh title="webhook_test.sh"
#!/bin/bash

#  Your access token and secret
ACCESS_TOKEN="your_api_key" 
SECRET="your_api_secret"

#  The data you want to send (for example, in JSON format)
PAYLOAD='{"key1": "value1", "key2": "value2"}'

#  Generate the HMAC signature
#  For security, Wandb includes the X-Wandb-Signature in the header computed 
#  from the payload and the shared secret key associated with the webhook 
#  using the HMAC with SHA-256 algorithm.
SIGNATURE=$(echo -n "$PAYLOAD" | openssl dgst -sha256 -hmac "$SECRET" -binary | base64)

#  Make the cURL request
curl -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "X-Wandb-Signature: $SIGNATURE" \
  -d "$PAYLOAD" API_ENDPOINT

```

  </TabItem>
</Tabs>

## Launchオートメーションを作成する
自動的にW&Bジョブを開始します。

:::info
このセクションでは、すでにジョブ、キューを作成しており、アクティブなエージェントが待機中であることを前提としています。詳細については、[W&B Launch ドキュメント](../launch/intro.md)を参照してください。
:::

1. **Event type**ドロップダウンからイベントタイプを選択します。サポートされているイベントについては[Event type](#event-types)セクションを参照してください。
2. (任意) **A new version is added to a registered model**イベントを選択した場合、**Registered model**ドロップダウンから登録モデルの名前を指定します。 
3. **Jobs**を**Action type**ドロップダウンから選択します。
4. **Job**ドロップダウンからW&B Launchジョブを選択します。
5. **Job version**ドロップダウンからバージョンを選択します。
6. (任意) 新しいジョブのためのハイパーパラメーターの上書きを指定します。
7. **Destination project**ドロップダウンからプロジェクトを選択します。
8. ジョブをキューに追加するキューを選択します。
9. **Next step**をクリックします。
10. **Automation name**フィールドにWebhookオートメーションの名前を入力します。
11. (任意) Webhookの説明を入力します。
12. **Create automation**ボタンをクリックします。

この例を参照してください[レポート](https://wandb.ai/examples/wandb_automations/reports/Model-CI-with-W-B-Automations--Vmlldzo0NDY5OTIx)で、W&B Launchを使用してモデルCIのためのオートメーションを作成する方法をご確認ください。

## オートメーションを表示する

W&B App UIから登録モデルに関連付けられたオートメーションを表示する。

1. [Model Registry App](https://wandb.ai/registry/model)に移動します。
2. 登録モデルを選択します。
3. ページの下部にスクロールして**Automations**セクションを見つけます。

Automationsセクションには、選択したモデルに対して作成されたオートメーションの次のプロパティが表示されます:

- **Trigger type**: 設定されたトリガーのタイプ。
- **Action type**: オートメーションをトリガーするアクションタイプ。利用可能なオプションはWebhooksとLaunchです。
- **Action name**: オートメーションを作成する際に指定したアクション名。
- **Queue**: ジョブがキューに追加されたキューの名前。このフィールドは、Webhookアクションタイプを選択した場合は空白のままです。

## オートメーションを削除する
オートメーションに関連付けられたモデルを削除します。アクションが完了する前にオートメーションを削除した場合でも、進行中のアクションには影響ありません。

