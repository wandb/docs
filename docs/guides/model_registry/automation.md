---
description: モデルCI（自動化モデルの評価パイプライン）およびモデルデプロイメントのためのオートメーションを使用します。
title: Model registry automations
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Triggering CI/CD events with model registry changes

モデルテストとデプロイの自動化を実行するためのワークフローステップをトリガーする自動化を作成します。自動化を作成するには、[イベントタイプ](#event-types)に基づいて発生する[アクション](#action-types)を定義します。

例えば、新しいバージョンのRegistered Modelを追加したときに、モデルをGitHubに自動的にデプロイするトリガーを作成できます。

:::note
カスタムモデルと新しいバージョンをW&B Model RegistryからAmazon SageMaker Endpointsに自動的にデプロイする方法のステップバイステップの説明は、[このYouTubeビデオ](https://www.youtube.com/watch?v=s5CMj_w3DaQ&ab_channel=Weights%26Biases)を参照してください。
:::

## Event types
*Event*とは、W&Bエコシステム内で発生する変更のことです。Model Registryは、**新しいArtifactをRegistered Modelにリンクする**イベントタイプと、**Registered Modelのバージョンに新しいエイリアスを追加する**イベントタイプの2種類をサポートしています。

:::tip
**新しいArtifactをRegistered Modelにリンクする**イベントタイプを使用して、新しいモデル候補をテストします。**Registered Modelのバージョンに新しいエイリアスを追加する**イベントタイプは、`deploy`のようなワークフローの特別なステップを表すエイリアスを指定するために使用し、新しいモデルバージョンにそのエイリアスが適用されるたびにトリガーされます。
:::

## Action types
アクションとは、トリガーの結果として発生する応答的な変化（内部または外部）のことです。Model Registryでは、[webhooks](#create-a-webhook-automation)と[W&B Launch Job](../launch/intro.md)の2種類のアクションを作成できます。

* Webhooks: HTTPリクエストを使用して、W&Bから外部のウェブサーバーと通信します。
* W&B Launch Job: [Jobs](../launch/create-launch-job.md)は再利用可能で構成可能なrunテンプレートであり、Kubernetes on EKSやAmazon SageMakerなどのデスクトップや外部の計算リソースで新しい[runs](../runs/intro.md)を迅速に起動できます。

以下のセクションでは、webhooksとW&B Launchを使用した自動化の作成方法について説明します。

## Create a webhook automation 
W&B App UIを使用してアクションに基づいたWebhookを自動化します。これを行うには、まずWebhookを設定し、その後にWebhook自動化を構成します。

:::info
エンドポイントにアドレスレコード（Aレコード）があるWebhookのエンドポイントを指定してください。W&Bは、`[0-255].[0-255].[0-255].[0.255]`のようにIPアドレスで直接公開されているエンドポイントや、`localhost`として公開されているエンドポイントへの接続をサポートしていません。この制限は、サーバーサイドリクエスト偽造（SSRF）攻撃やその他関連する脅威ベクトルから保護するためです。
:::

### Add a secret for authentication or authorization
シークレットはチームレベルの変数であり、資格情報、APIキー、パスワード、トークンなどのプライベートな文字列を隠すために使用します。W&Bは、シークレットを使用して平文内容を保護したい文字列を保存することを推奨します。

Webhookでシークレットを使用するには、まずそのシークレットをチームのシークレットマネージャーに追加する必要があります。

:::info
* W&Bの管理者のみがシークレットを作成、編集、削除できます。
* HTTP POSTリクエストを送信する外部サーバーがシークレットを使用しない場合、このセクションをスキップします。
* Azure、GCP、またはAWSデプロイメントで[W&B Server](../hosting/intro.md)を使用する場合、シークレットは利用可能です。異なるデプロイメントタイプを使用している場合でも、W&Bアカウントチームと連絡を取り、W&Bでシークレットを使用する方法について話し合ってください。
:::

Webhook自動化を使用する際にW&Bが提案するシークレットは次の2種類です：

* **アクセストークン**: 送信者を認証してWebhookリクエストをセキュリティ保護。
* **シークレット**: ペイロードから送信されるデータの真正性と整合性を確保。

Webhookを作成するには、以下の手順に従います：

1. W&B App UIに移動します。
2. **Team Settings**をクリックします。
3. ページを下にスクロールして**Team secrets**セクションを見つけます。
4. **New secret**ボタンをクリックします。
5. モーダルが表示されます。**Secret name**フィールドにシークレットの名前を入力します。
6. **Secret**フィールドにシークレットを追加します。
7. （オプション）Webhookが追加のシークレットキーやトークンを必要とする場合は、手順5と6を繰り返して他のシークレット（アクセストークンなど）を作成します。

Webhookを構成する際に使用するシークレットを指定します。詳細は[Webhookの構成](#configure-a-webhook)セクションを参照してください。

:::tip
シークレットを作成すると、W&Bのワークフローで `$` を使用してそのシークレットにアクセスできます。
:::

:::caution
W&B Serverでシークレットを使用する場合の考慮事項：

セキュリティニーズに対応するためのセキュリティ対策を構成する責任があります。

W&Bは、AWS、GCP、またはAzureのクラウドシークレットマネージャー内のW&Bインスタンスにシークレットを保存することを強く推奨します。AWS、GCP、Azureが提供するシークレットマネージャーは、高度なセキュリティ機能を備えています。

W&Bは、シークレットストアのバックエンドとしてKubernetesクラスターを使用することを推奨しません。Kubernetesクラスターは、クラウドシークレットマネージャー（AWS、GCP、Azure）のW&Bインスタンスを使用できない場合で、かつクラスターを使用する際のセキュリティ脆弱性を防ぐ方法を理解している場合にのみ検討してください。
:::

### Configure a webhook
Webhookを使用する前に、まずW&B App UIでそのWebhookを構成する必要があります。

:::info
* W&Bの管理者のみがW&B Team用にWebhookを構成できます。
* Webhookが追加のシークレットキーやトークンを必要とする場合、既に[1つ以上のシークレットを作成しました](#add-a-secret-for-authentication-or-authorization)ことを確認してください。
:::

1. W&B App UIに移動します。
2. **Team Settings**をクリックします。
3. ページを下にスクロールして**Webhooks**セクションを見つけます。
4. **New webhook**ボタンをクリックします。
5. **Name**フィールドにWebhookの名前を入力します。
6. **URL**フィールドにWebhookのエンドポイントURLを入力します。
7. （オプション）**Secret**ドロップダウンメニューから、Webhookペイロードを認証するために使用するシークレットを選択します。
8. （オプション）**Access token**ドロップダウンメニューから、送信者を認可するために使用するアクセストークンを選択します。
9. （オプション）**Access token**ドロップダウンメニューから、Webhookを認証するために必要な追加のシークレットキーやトークン（アクセストークンなど）を選択します。

:::note
Webhookペイロードの`event_type`キーは、GitHubのワークフローYAMLファイルの`types`フィールドと一致する必要があります。
:::

### Add a webhook
Webhookが構成され、（オプション）シークレットが設定されたら、[https://wandb.ai/registry/model](https://wandb.ai/registry/model)のModel Registry Appに移動します。

1. **Event type**ドロップダウンから[イベントタイプ](#event-types)を選択します。
![](/images/models/webhook_select_event.png)
2. （オプション）**新しいバージョンがRegistered Modelに追加された場合**イベントを選択した場合、**Registered model**ドロップダウンからRegistered Modelの名前を入力します。
![](/images/models/webhook_new_version_reg_model.png)
3. **Action type**ドロップダウンから**Webhooks**を選択します。
4. **Next step**ボタンをクリックします。
5. **Webhook**ドロップダウンからWebhookを選択します。
![](/images/models/webhooks_select_from_dropdown.png)
6. （オプション）JSON表記エディタにペイロードを入力します。一般的なユースケースの例は[例のペイロード](#example-payloads)セクションを参照してください。
7. **Next step**をクリックします。
8. **Automation name**フィールドにWebhook自動化の名前を入力します。
![](/images/models/webhook_name_automation.png)
9. （オプション）Webhookの説明を入力します。
10. **Create automation**ボタンをクリックします。

### Example payloads

以下のタブには、一般的なユースケースに基づく例のペイロードが示されています。例内では、ペイロードパラメータの条件オブジェクトを参照するために次のキーを使用しています：
* `${event_type}` アクションをトリガーしたイベントのタイプを指します。
* `${event_author}` アクションをトリガーしたユーザーを指します。
* `${artifact_version}` アクションをトリガーした特定のArtifactバージョンを指します。Artifactインスタンスとして渡されます。
* `${artifact_version_string}` アクションをトリガーした特定のArtifactバージョンを指します。文字列として渡されます。
* `${artifact_collection_name}` ArtifactバージョンがリンクされているArtifactコレクションの名前を指します。
* `${project_name}` アクションをトリガーしたプロジェクトの名前を指します。
* `${entity_name}` アクションをトリガーしたエンティティの名前を指します。

<Tabs
  defaultValue="github"
  values={[
    {label: 'GitHub repository dispatch', value: 'github'},
    {label: 'Microsoft Teams notification', value: 'microsoft'},
    {label: 'Slack notifications', value: 'slack'},
  ]}>
  <TabItem value="github">

:::info
アクセストークンが必要な権限を持っているか確認して、GHAワークフローをトリガーしてください。詳細は[GitHubのドキュメント](https://docs.github.com/en/rest/repos/repos?#create-a-repository-dispatch-event)を参照してください。
:::
  
W&Bからリポジトリディスパッチを送信してGitHubアクションをトリガーします。例えば、リポジトリディスパッチを`on`キーのトリガーとして受け付けるワークフローがあるとします：

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
Webhookペイロード内の`event_type`キーはGitHubワークフローYAMLファイルの`types`フィールドと一致する必要があります。
:::

テンプレート文字列の内容と位置は、イベントまたはモデルバージョンの自動化が構成されている場合によって異なります。`${event_type}`は「LINK_ARTIFACT」または「ADD_ARTIFACT_ALIAS」としてレンダリングされます。以下に例のマッピングを示します：

  ```json
  ${event_type} --> "LINK_ARTIFACT" または "ADD_ARTIFACT_ALIAS"
  ${event_author} --> "<wandb-user>"
  ${artifact_version} --> "wandb-artifact://_id/QXJ0aWZhY3Q6NTE3ODg5ODg3"
  ${artifact_version_string} --> "<entity>/model-registry/<registered_model_name>:<alias>"
  ${artifact_collection_name} --> "<registered_model_name>"
  ${project_name} --> "model-registry"
  ${entity_name} --> "<entity>"
  ```

テンプレート文字列を使用して、W&BからGitHub Actionsやその他のツールにコンテキストを動的に渡すことができます。これらのツールがPythonスクリプトを呼び出すことができれば、Registered Model Artifactsを[W&B API](../artifacts/download-and-use-an-artifact.md)を通じて消費することができます。

リポジトリディスパッチの詳細は、[GitHubマーケットプレイスの公式ドキュメント](https://github.com/marketplace/actions/repository-dispatch)を参照してください。

Model EvaluationのWebhook Automationsについての詳細は[Webhook Automations for Model Evaluation](https://www.youtube.com/watch?v=7j-Mtbo-E74&ab_channel=Weights%26Biases)と、Model DeploymentのWebhook Automationsについての詳細は[Webhook Automations for Model Deployment](https://www.youtube.com/watch?v=g5UiAFjM2nA&ab_channel=Weights%26Biases)のYouTubeビデオを参照してください。

Model CIでのGitHub Actions Webhook Automationの使用方法については、このW&Bの[レポート](https://wandb.ai/wandb/wandb-model-cicd/reports/Model-CI-CD-with-W-B--Vmlldzo0OTcwNDQw)を参照してください。Modal LabsのWebhookでModel CIを作成する方法については、この[GitHubリポジトリ](https://github.com/hamelsmu/wandb-modal-webhook)を参照してください。

  </TabItem>
  <TabItem value="microsoft">

Teams Channel用のWebhook URLを取得するために「Incoming Webhook」を設定します。以下は例のペイロードです：
  
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
  あなたのペイロードに実行時にW&Bデータを注入するためにテンプレート文字列を使用できます。（上記のTeamsの例のように）

  </TabItem>
  <TabItem value="slack">

Slackアプリを設定し、[Slack APIドキュメント](https://api.slack.com/messaging/webhooks)で説明されている手順に従って、インカミングWebhookの統合を追加します。`Bot User OAuth Token`の下で指定されているシークレットをW&B Webhookのアクセストークンとして指定していることを確認してください。

以下は例のペイロードです：

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

### Troubleshoot your webhook

W&B App UIを使用して対話的にWebhookをトラブルシューティングするか、Bashスクリプトを使用してプログラムでトラブルシューティングします。新しいWebhookを作成する際や既存のWebhookを編集する際にWebhookをトラブルシューティングできます。

<Tabs
  defaultValue="app"
  values={[
    {label: 'W&B App UI', value: 'app'},
    {label: 'Bash script', value: 'bash'},
  ]}>
  <TabItem value="app">

W&B App UIを使用してWebhookを対話的にテストします。

1. W&BのTeam Settingsページに移動します。
2. **Webhooks