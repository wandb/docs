---
description: プロジェクトスコープのアーティファクト自動化を使用して、アーティファクトコレクション内でエイリアスやバージョンが作成または変更されたときにアクションをトリガーします。
title: Artifact automations
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Artifactの変更でCI/CDイベントをトリガーする

Artifactが変更されたときにトリガーされる自動化を作成します。Artifactのバージョン管理のために下流のアクションを自動化したい場合にArtifactの自動化を使用します。自動化を作成するには、発生させたい[action](#action-types)を[event type](#event-types)に基づいて定義します。

Artifactの変更からトリガーされる自動化の一般的なユースケースには以下のものがあります：

* 新しいバージョンの評価/ホールドアウトデータセットがアップロードされたときに、モデルレジストリ内の最良のトレーニングモデルを使用して推論を行い、パフォーマンス情報を含むレポートを作成する[launch job](#create-a-launch-automation)をトリガーする。
* トレーニングデータセットの新しいバージョンが「プロダクション」とラベル付けされたときに、現在の最良パフォーマンスモデルの設定で再トレーニングlaunch jobをトリガーする。

:::info
Artifactの自動化はプロジェクトにスコープされています。これは、プロジェクト内のイベントのみがArtifact自動化をトリガーすることを意味します。

これはW&B Model Registryで作成された自動化とは対照的です。model registryで作成された自動化は、Model Registryにスコープされています。これらは、[Model Registry](../model_registry/intro.md)にリンクされたモデルバージョンでイベントが実行されたときにトリガーされます。モデルバージョンの自動化を作成する方法については、[Automations for Model CI/CD](../model_registry/automation.md)ページを参照してください。
:::

## Event types
*Event*はW&Bエコシステム内で発生する変更です。プロジェクトのArtifactコレクションに対して、2つの異なるイベントタイプを定義できます：**Artifactの新しいバージョンがコレクションに作成される**と**Artifactエイリアスが追加される**。

:::tip
Artifactのバージョンごとに定期的なアクションを適用するには、**Artifactの新しいバージョンがコレクションに作成される**イベントタイプを使用します。例えば、新しいデータセットArtifactバージョンが作成されたときに自動的にトレーニングジョブを開始する自動化を作成できます。

特定のエイリアスがArtifactバージョンに適用されたときにアクティブになる自動化を作成するには、**Artifactエイリアスが追加される**イベントタイプを使用します。例えば、「test-set-quality-check」エイリアスをArtifactに追加することで、そのデータセットでの下流プロセシングをトリガーするアクションをトリガーする自動化を作成できます。
:::

## Action types
アクションは、あるトリガーの結果として発生する応答的な変化（内部または外部）です。プロジェクトのArtifactコレクションのイベントに応じて作成できるアクションには、webhooksと[W&B Launch Jobs](../launch/intro.md)の2種類があります。

* Webhooks: W&BからHTTPリクエストで外部ウェブサーバーと通信します。
* W&B Launch Job: [Jobs](../launch/create-launch-job.md)は再利用可能で設定可能なrunテンプレートで、デスクトップでローカルに新しい[runs](../runs/intro.md)を素早く開始したり、EKS上のKubernetesやAmazon SageMakerなどの外部コンピュートリソースを使用したりすることができます。

以下のセクションでは、webhooksとW&B Launchを使用して自動化を作成する方法について説明します。

## Webhook automationを作成する
W&B App UIを使用して、アクションに基づいてwebhookを自動化します。これを行うには、まずwebhookを確立し、その後webhookの自動化を設定します。

:::info
W&Bでは、IPアドレス（例えば`[0-255].[0-255].[0-255].[0-255]`）や`localhost`として公開されているエンドポイントへの接続はサポートしていません。エンドポイントにはアドレスレコード（Aレコード）を指定してください。この制限は、サーバーサイドリクエストフォージェリ（SSRF）攻撃やその他関連する脅威ベクトルから保護するのに役立ちます。
:::

### 認証または承認のためのシークレットを追加する
シークレットは、資格情報、APIキー、パスワード、トークンなどのプライベート文字列を隠蔽するためのチームレベルの変数です。W&Bでは、保存したい任意の文字列を保護するためにシークレットを使用することを推奨しています。

webhookでシークレットを使用するには、まずそのシークレットをチームのシークレットマネージャーに追加する必要があります。

:::info
* W&Bの管理者のみがシークレットを作成、編集、削除できます。
* リクエストを送信する外部サーバーがシークレットを使用しない場合は、このセクションをスキップしてください。
* Azure、GCP、AWSデプロイメントで[W&B Server](../hosting/intro.md)を使用している場合もシークレットは利用可能です。異なるデプロイメントタイプを使用している場合は、W&Bアカウントチームと連絡を取り、W&Bでシークレットを使用する方法について相談してください。
:::

Webhook automationを使用する際に作成を推奨するシークレットには、以下の2種類があります：

* **Access tokens**: Webhookリクエストのセキュリティを確保するために送信者を承認します。
* **Secret**: ペイロードから送信されたデータの真正性と完全性を確保します。

以下の手順に従ってwebhookを作成します：

1. W&B App UIに移動します。
2. **Team Settings**をクリックします。
3. ページを下にスクロールして**Team secrets**セクションを見つけます。
4. **New secret**ボタンをクリックします。
5. モーダルが表示されます。**Secret name**フィールドにシークレットの名前を入力します。
6. **Secret**フィールドにシークレットを追加します。
7. （オプション）必要に応じて、webhookを認証するために追加のシークレットキーやトークン（例えばアクセスToken）を作成する場合は、5と6の手順を繰り返します。

Webhookを設定する際に使用したいシークレットを指定します。詳細は[Webhookを設定する](#configure-a-webhook)セクションを参照してください。

:::tip
シークレットを作成すると、そのシークレットをW&Bワークフロー内で `$` を使ってアクセスできます。
:::

### Webhookを設定する
Webhookを使用する前に、W&B App UIでそのWebhookを設定する必要があります。

:::info
* W&Bの管理者のみがW&BチームのためにWebhookを設定できます。
* Webhookを認証するために追加のシークレットキーやトークンが必要な場合は、先に[シークレットを作成](#add-a-secret-for-authentication-or-authorization)してください。
:::

1. W&B App UIに移動します。
2. **Team Settings**をクリックします。
4. ページを下にスクロールして**Webhooks**セクションを見つけます。
5. **New webhook**ボタンをクリックします。
6. **Name**フィールドにWebhookの名前を入力します。
7. **URL**フィールドにWebhookのエンドポイントURLを入力します。
8. （オプション） **Secret**ドロップダウンメニューから、Webhookペイロードを認証するために使用するシークレットを選択します。
9. （オプション） **Access token**ドロップダウンメニューから、送信者を認証するために使用するアクセスTokenを選択します。
9. （オプション） **Access token** ドロップダウンメニューから、Webhookを認証するのに必要な追加のシークレットキーやトークン（例えばアクセスToken）を選択します。

:::note
POSTリクエストでシークレットとアクセスTokenが指定される場所を確認するには、[Webhookのトラブルシューティング](#troubleshoot-your-webhook)セクションを参照してください。
:::

### Webhookを追加する
Webhookが設定され、（オプションで）シークレットが作成されたら、プロジェクトワークスペースに移動します。左側のサイドバーにある**Automations**タブをクリックします。

1. **Event type**ドロップダウンから[event type](#event-types)を選択します。
![](/images/artifacts/artifact_webhook_select_event.png)
2. **A new version of an artifact is created in a collection**イベントを選択した場合、自動化が対応するArtifactコレクションの名前を**Artifact collection**ドロップダウンから提供します。 
![](/images/artifacts/webhook_new_version_artifact.png)
3. **Action type**ドロップダウンから**Webhooks**を選択します。
4. **Next step**ボタンをクリックします。
5. **Webhook**ドロップダウンからWebhookを選択します。
![](/images/artifacts/artifacts_webhooks_select_from_dropdown.png)
6. （オプション）JSON表現エディタにペイロードを提供します。一般的なユースケースの例については、[Example payload](#example-payloads)セクションを参照してください。
7. **Next step**をクリックします。
8. **Automation name**フィールドにWebhook自動化の名前を入力します。
![](/images/artifacts/artifacts_webhook_name_automation.png)
9. （オプション）Webhookの説明を提供します。
10. **Create automation**ボタンをクリックします。

### Example payloads

以下のタブは一般的なユースケースに基づいたペイロードの例を示します。これらの例の中では、ペイロードパラメータ内の条件オブジェクトを指す以下のキーを参照しています：
* `${event_type}` イベントをトリガーしたアクションタイプを指します。
* `${event_author}` アクションをトリガーしたユーザーを指します。
* `${artifact_version}` アクションをトリガーした特定のArtifactバージョンを指します。artifactインスタンスとして渡されます。
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
アクセスTokensがGHAワークフローをトリガーするために必要な権限セットを持っていることを確認してください。詳細は、[GitHub Docs](https://docs.github.com/en/rest/repos/repos?#create-a-repository-dispatch-event)を参照してください。
:::

  W&BからGitHubアクションをトリガーするためにレポジトリディスパッチを送信します。例えば、`on`キーのトリガーとしてレポジトリディスパッチを受け入れるワークフローがあるとしましょう：

  ```yaml
  on:
    repository_dispatch:
      types: BUILD_AND_DEPLOY
  ```

  レポジトリのペイロードは次のようになるかもしれません：

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
Webhookペイロードの`event_type`キーは、GitHubワークフローYAMLファイルの`types`フィールドと一致している必要があります。
:::

  レンダリングされたテンプレート文字列の内容と位置は、イベントまたは自動化が設定されたモデルバージョンに依存します。`${event_type}`は「LINK_ARTIFACT」または「ADD_ARTIFACT_ALIAS」としてレンダリングされます。以下は例のマッピングです：

  ```json
  ${event_type} --> "LINK_ARTIFACT"または"ADD_ARTIFACT_ALIAS"
  ${event_author} --> "<wandb-user>"
  ${artifact_version} --> "wandb-artifact://_id/QXJ0aWZhY3Q6NTE3ODg5ODg3"
  ${artifact_version_string} --> "<entity>/<project_name>/<artifact_name>:<alias>"
  ${artifact_collection_name} --> "<artifact_collection_name>"
  ${project_name} --> "<project_name>"
  ${entity_name} --> "<entity>"
  ```

  テンプレート文字列を使用して、W&BからGitHub Actionsや他のツールにコンテキストを動的に渡します。これらのツールがPythonスクリプトを呼び出すことができるなら、[W&B API](../artifacts/download-and-use-an-artifact.md)を通じてW&B Artifactsを消費できます。

  レポジトリディスパッチの詳細については、[GitHub Marketplaceの公式ドキュメント](https://github.com/marketplace/actions/repository-dispatch)を参照してください。

  </TabItem>
  <TabItem value="microsoft">

  Teams Channelの設定を行い、Webhook URLを取得するための「Incoming Webhook」を設定します。以下はペイロードの例です：

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
  上記のTeams例に示されているように、テンプレート文字列を使用して実行時にW&Bデータをペイロードに注入できます。

  </TabItem>
  <TabItem value="slack">

  Slackアプリをセットアップし、[Slack APIドキュメント](https://api.slack.com/messaging/webhooks)に記載されている手順に従い、インカミングWebhookのインテグレーションを追加します。W&B webhookのアクセスTokenとして、`Bot User OAuth Token`に指定されたシークレットがあることを確認してください。

  以下はペイロードの例です：

  ```json
    {
        "text": "New alert from WANDB!",
    "blocks": [
        {
                "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "Artifact event: ${event_type}"
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

W&B App UIまたはBashスクリプトでインタラクティブにWebhookのトラブルシューティングを行います。新しいWebhookを作成するか、既存のWebhookを編集する際にWebhookのトラブルシューティングが可能です。

<Tabs
  defaultValue="app"
  values={[
    {label: 'W&B App UI', value: 'app'},
    {label: 'Bash script', value: 'bash'},
  ]}>
  <TabItem value="app">

W&B App UIを使用してインタラクティブにWebhookをテストします。

1. W&Bチーム設定ページに移動します。
2. **Webhooks**セクションまでスクロールします。
3. Webhookの名前の横にある水平の3点（meatballアイコン）をクリックします。
4. **Test**を選択します。
5. 表示されるUIパネルから、POSTリクエストをフィールドに貼り付けます。
![](/images/models/webhook_ui.png)
6. **Test webhook**をクリックします。

W&B App UI内で、エンドポイントによって行われた