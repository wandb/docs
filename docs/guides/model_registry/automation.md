---
description: モデルCI（自動化されたモデルの評価パイプライン）とモデルデプロイメントにオートメーションを使用します。
title: Model registry automations
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# モデルレジストリの変更によるCI/CDイベントのトリガー

モデルの自動テストやデプロイメントなどのワークフローステップをトリガーするオートメーションを作成します。オートメーションを作成するには、[イベントタイプ](#event-types)に基づいて発生させたい[アクション](#action-types)を定義します。

例えば、新しいバージョンの登録モデルを追加したときにモデルをGitHubに自動デプロイするトリガーを作成できます。

:::note
この [YouTube動画](https://www.youtube.com/watch?v=s5CMj_w3DaQ&ab_channel=Weights%26Biases) で、カスタムモデルおよび新しいモデルバージョンをW&BモデルレジストリからAmazon SageMaker Endpointsに自動デプロイする手順をステップバイステップでご覧ください。
:::

## イベントタイプ
*イベント*とは、W&Bエコシステム内で行われる変更のことです。モデルレジストリは、**新しいアーティファクトを登録モデルにリンクする** と **登録モデルのバージョンに新しいエイリアスを追加する** の2つのイベントタイプをサポートしています。

:::tip
新しいモデル候補をテストするには、**新しいアーティファクトを登録モデルにリンクする** イベントタイプを使用します。ワークフローの特別なステップ（例：`deploy`）を表すエイリアスを指定するには、**登録モデルのバージョンに新しいエイリアスを追加する** イベントタイプを使用し、そのエイリアスを持つ新しいモデルバージョンが適用されるたびにこのイベントが発生します。
:::

## アクションタイプ
アクションとは、トリガーの結果として発生する内部または外部の変異です。モデルレジストリでは、[webhooks](#create-a-webhook-automation) と [W&B Launch Jobs](../launch/intro.md) の2種類のアクションを作成できます。

* Webhooks: W&BからHTTPリクエストを介して外部のウェブサーバーと通信します。
* W&B Launchジョブ: [Jobs](../launch/create-launch-job.md) は、デスクトップ上やKubernetes on EKS、Amazon SageMakerなど外部の計算リソースで新しい[run](../runs/intro.md)を迅速に起動できる再利用可能で設定可能なrunテンプレートです。

以下のセクションでは、webhooksとW&B Launchを使用したオートメーションの作成方法について説明します。

## Webhookオートメーションを作成
W&B App UIを使ってアクションに基づくwebhookを自動化します。これを行うには、まずwebhookを設定し、次にwebhookオートメーションを構成します。

:::info
webhookのエンドポイントにはアドレスレコード(Aレコード)を指定してください。W&Bは、`[0-255].[0-255].[0-255].[0.255]`のように直接IPアドレスで公開されているエンドポイントや`localhost`として公開されているエンドポイントへの接続をサポートしていません。この制限は、サーバー側リクエスト偽装(SSRF)攻撃やその他の関連する脅威ベクターから保護するためです。
:::

### 認証や認可のためのシークレットを追加
シークレットは、資格情報、APIキー、パスワード、トークンなどのプライベートな文字列を隠すためのチームレベルの変数です。W&Bは、平文の内容を保護したい文字列を保存するためにシークレットを使用することを推奨します。

webhookでシークレットを使用するには、まずそのシークレットをチームのシークレットマネージャに追加する必要があります。

:::info
* シークレットを作成、編集、削除できるのはW&B管理者のみです。
* HTTP POSTリクエストを送信する外部サーバーでシークレットを使用していない場合、このセクションはスキップしてください。
* [W&B Server](../hosting/intro.md) をAzure, GCP, またはAWSデプロイメントで使用している場合もシークレットを利用できます。異なるデプロイメントタイプを使用している場合、W&Bアカウントチームに連絡してシークレットの使用方法について相談してください。
:::

W&Bがwebhookオートメーションを使用するときに作成を推奨するシークレットには次の2種類があります：

* **トークンへのアクセス**: 送信者を認可してwebhookリクエストを安全に扱うため
* **シークレット**: ペイロードから送信されるデータの信憑性と整合性を保証するため

以下の手順に従ってwebhookを作成してください：

1. W&B App UIに移動します。
2. **Team Settings**をクリックします。
3. ページをスクロールダウンし、**Team secrets**セクションを見つけます。
4. **New secret**ボタンをクリックします。
5. モーダルが表示されます。**Secret name**フィールドにシークレットの名前を入力します。
6. **Secret**フィールドにシークレットを追加します。
7. (オプション）webhookに追加のシークレットキーやトークンが必要な場合、ステップ5と6を繰り返して別のシークレット（例えば認証トークン）を作成します。

webhookを構成する際に使用するシークレットを指定します。詳細は[Webhookを構成する](#configure-a-webhook)セクションを参照してください。

:::tip
シークレットを一度作成すると、W&Bのワークフロー内でそのシークレットに`$`を使ってアクセスできます。
:::

:::caution
W&B Serverでシークレットを使用する場合の考慮点：

自分のセキュリティニーズを満たすためのセキュリティ対策を構成する責任があります。

W&Bは、シークレットをAWS、GCP、またはAzureによって提供されているクラウドシークレットマネージャーのW&Bインスタンスに保存することを強く推奨します。AWS、GCP、およびAzureによって提供されるシークレットマネージャーは、高度なセキュリティ機能を備えています。

シークレットストアのバックエンドとしてKubernetesクラスターを使用することはお勧めしません。Kubernetesクラスターを使用する場合、クラスタ使用によるセキュリティ脆弱性を防ぐ方法を理解していることが必要です。
:::

### Webhookを構成
webhookを使用する前に、まずW&B App UIでそのwebhookを構成する必要があります。

:::info
* W&B TeamのW&B管理者のみがwebhookを構成できます。
* webhookの認証に追加のシークレットキーやトークンが必要な場合、既に[1つ以上のシークレットを作成した](#add-a-secret-for-authentication-or-authorization)ことを確認してください。
:::

1. W&B App UIに移動します。
2. **Team Settings**をクリックします。
4. ページをスクロールダウンし、**Webhooks**セクションを見つけます。
5. **New webhook**ボタンをクリックします。
6. **Name**フィールドにwebhookの名前を入力します。
7. **URL**フィールドにwebhookのエンドポイントURLを入力します。
8. (オプション) **Secret** ドロップダウンメニューから、webhookペイロードの認証に使用するシークレットを選択します。
9. (オプション) **Access token** ドロップダウンメニューから、送信者を認可するアクセス トークンを選択します。
9. (オプション) **Access token** ドロップダウンメニューから、webhookの認証に必要な追加のシークレットキーまたはトークン（例えば認証トークン）を選択します。

:::note
POSTリクエストでシークレットと認証トークンがどこに指定されているかについては、[Webhookのトラブルシューティング](#troubleshoot-your-webhook)セクションを参照してください。
:::

### Webhookを追加
webhookを構成し、（オプションで）シークレットを設定したら、[https://wandb.ai/registry/model](https://wandb.ai/registry/model)のモデルレジストリアプリに移動します。

1. **Event type** ドロップダウンから、[イベントタイプ](#event-types)を選択します。
![](/images/models/webhook_select_event.png)
2. (オプション) **A new version is added to a registered model** イベントを選択した場合、**Registered model** ドロップダウンから登録モデルの名前を指定します。
![](/images/models/webhook_new_version_reg_model.png)
3. **Action type** ドロップダウンから**Webhooks**を選択します。
4. **Next step**ボタンをクリックします。
5. **Webhook** ドロップダウンからwebhookを選択します。
![](/images/models/webhooks_select_from_dropdown.png)
6. (オプション) JSONエディタにペイロードを入力します。一般的なユースケースの例については[Example payload](#example-payloads)セクションを参照してください。
7. **Next step**をクリックします。
8. **Automation name** フィールドにwebhookオートメーションの名前を入力します。
![](/images/models/webhook_name_automation.png)
9. (オプション) webhookの説明を入力します。
10. **Create automation** ボタンをクリックします。

### Example payloads

以下のタブでは、一般的なユースケースに基づいたペイロードの例を示しています。これらの例では、ペイロードパラメータ内の条件オブジェクトを参照するための以下のキーが使用されています：
* `${event_type}` トリガーされたアクションのイベントタイプを参照
* `${event_author}` トリガーされたアクションのユーザーを参照
* `${artifact_version}` トリガーされたアクションの特定のアーティファクトバージョンを参照。アーティファクトインスタンスとして渡されます。
* `${artifact_version_string}` トリガーされたアクションの特定のアーティファクトバージョンを参照。文字列として渡されます。
* `${artifact_collection_name}` アーティファクトバージョンがリンクされているアーティファクトコレクションの名前を参照
* `${project_name}` トリガーされたアクションの変更を所有するプロジェクトの名前を参照
* `${entity_name}` トリガーされたアクションの変更を所有するエンティティの名前を参照

<Tabs
  defaultValue="github"
  values={[
    {label: 'GitHub repository dispatch', value: 'github'},
    {label: 'Microsoft Teams notification', value: 'microsoft'},
    {label: 'Slack notifications', value: 'slack'},
  ]}>
  <TabItem value="github">

:::info
アクセス トークンにGHAワークフローをトリガーするために必要な権限が設定されていることを確認してください。 詳細については、[こちらのGitHub Docs](https://docs.github.com/en/rest/repos/repos?#create-a-repository-dispatch-event)をご覧ください。
:::

W&Bからリポジトリ ディスパッチを送信して、GitHubアクションをトリガーします。例えば、`on`キーのトリガーとしてリポジトリ ディスパッチを受け入れるワークフローがあるとします：

  ```yaml
  on:
    repository_dispatch:
      types: BUILD_AND_DEPLY
  ```

リポジトリのペイロードは以下のようになります：

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
webhookペイロードの`event_type`キーは、GitHubワークフローファイルの`types`フィールドと一致する必要があります。
:::

テンプレート文字列の内容と位置は、オートメーションが構成されているイベントまたはモデルバージョンに依存します。`${event_type}` は"LINK_ARTIFACT"または"ADD_ARTIFACT_ALIAS"としてレンダーされます。以下に例のマッピングを示します：

  ```json
  ${event_type} --> "LINK_ARTIFACT" または "ADD_ARTIFACT_ALIAS"
  ${event_author} --> "<wandb-user>"
  ${artifact_version} --> "wandb-artifact://_id/QXJ0aWZhY3Q6NTE3ODg5ODg3"
  ${artifact_version_string} --> "<entity>/model-registry/<registered_model_name>:<alias>"
  ${artifact_collection_name} --> "<registered_model_name>"
  ${project_name} --> "model-registry"
  ${entity_name} --> "<entity>"
  ```

テンプレート文字列を使用して、W&BからGitHub Actionsやその他のツールにコンテキストを動的に渡します。これらのツールがPythonスクリプトを呼び出すことができれば、[W&B API](../artifacts/download-and-use-an-artifact.md)を通じて登録モデルアーティファクトを消費できます。

リポジトリ ディスパッチの詳細については、 [GitHubマーケットプレイスの公式ドキュメント](https://github.com/marketplace/actions/repository-dispatch)を参照してください。

[モデル評価のためのWebhookオートメーション](https://www.youtube.com/watch?v=7j-Mtbo-E74&ab_channel=Weights%26Biases) と [モデルデプロイメントのためのWebhookオートメーション](https://www.youtube.com/watch?v=g5UiAFjM2nA&ab_channel=Weights%26Biases) についてのステップバイステップのYouTube動画をご覧ください。

GitHub Actions webhookオートメーションを使用したModel CIの使い方については、こちらのW&B [Report](https://wandb.ai/wandb/wandb-model-cicd/reports/Model-CI-CD-with-W-B--Vmlldzo0OTcwNDQw)を参照してください。Modal Labs webhookを使用してModel CIを作成する方法については、 [このGitHubリポジトリ](https://github.com/hamelsmu/wandb-modal-webhook)を参照してください。

  </TabItem>
  <TabItem value="microsoft">

Teams ChannelのWebhook URLを取得するために、「Incoming Webhook」を設定します。以下は例のペイロードです：

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
実行時にW&Bデータをペイロードに挿入するために、テンプレート文字列を使用できます（上記のTeams例のように）。

  </TabItem>
  <TabItem value="slack">

あなたのSlackアプリをセットアップし、[Slack API ドキュメント](https://api.slack.com/messaging/webhooks)に記載されている手順に従って、インカミングWebhookインテグレーションを追加します。[`Bot User OAuthトークン`]として指定されたシークレットをW&B webhookのアクセス トークンとして確保してください。

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

### Webhook のトラブルシューティング

W&B アプリケーション の UI を使用して対話的に Webhook のトラブルシューティングを行うか、または Bash スクリプトを使用してプログラム的にトラブルシューティングを行います。新しい Webhook を作成する際や既存の Webhook を編集する際に、Webhook のトラブルシューティングを行うことができます。

<Tabs
  defaultValue="app"
  values={[
    {label: 'W&B アプリケーション UI', value: 'app'},
    {label: 'Bash script', value: 'bash'},
  ]}>
  <TabItem value="app">

W&B アプリケーション UI を使用して Webhook を対話的にテストします。

1. W&B Team Settings ページに移動します。
2. **Webhooks** セクションまでスクロールします。
3. Webhook の名前の横にある水平の三点（ミートボールアイコン）をクリックします。
4. **Test** を選択します。
5. 表示される UI パネルから、POST リクエストを表示されるフィールドに貼り付けます。
![](/images/models/webhook_ui.png)
6. **Test webhook** をクリックします。

W&B アプリケーション UI 内では、エンドポイントによって生成されたレスポンスが投稿されます。

![](/images/models/webhook_ui_testing.gif)

実際の例については、[Weights & Biases における Webhook のテスト](https://www.youtube.com/watch?v=bl44fDpMGJw&ab_channel=Weights%26Biases)の YouTube ビデオをご覧ください。

  </TabItem>
  <TabItem value="bash">

以下の bash スクリプトは、W&B が Webhook 自動化をトリガーしたときに送信する POST リクエストに類似した POST リクエストを生成します。

以下のコードをシェルスクリプトにコピー＆ペーストして、Webhook のトラブルシューティングを行います。以下の値を自分のものに置き換えてください。

* `ACCESS_TOKEN`
* `SECRET`
* `PAYLOAD`
* `API_ENDPOINT`

```sh title="webhook_test.sh"
#!/bin/bash

# アクセストークンとシークレット
ACCESS_TOKEN="your_api_key" 
SECRET="your_api_secret"

# 送信したいデータ（例：JSON形式）
PAYLOAD='{"key1": "value1", "key2": "value2"}'

# HMAC シグネチャを生成
# セキュリティのために、Wandb はヘッダに X-Wandb-Signature を含め
# ペイロードと Webhook に関連付けられた共有シークレット キーから
# HMAC with SHA-256 アルゴリズムを使用して計算します。
SIGNATURE=$(echo -n "$PAYLOAD" | openssl dgst -sha256 -hmac "$SECRET" -binary | base64)

# cURL リクエストの作成
curl -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "X-Wandb-Signature: $SIGNATURE" \
  -d "$PAYLOAD" API_ENDPOINT
```

  </TabItem>
</Tabs>

## Launch 自動化を作成する

自動的に W&B ジョブを開始します。

:::info
このセクションは、すでにジョブ、キューが作成されており、アクティブなエージェントがポーリングしていることを前提としています。詳細については、[W&B Launch docs](../launch/intro.md)をご覧ください。
:::

1. **Event type** ドロップダウンからイベントタイプを選択します。サポートされているイベントについては [Event type](#event-types) セクションを参照してください。
2. （オプション）**A new version is added to a registered model** イベントを選択した場合、**Registered model** ドロップダウンから登録されたモデルの名前を提供します。
3. **Action type** ドロップダウンから **Jobs** を選択します。
4. **Job** ドロップダウンから W&B Launch ジョブを選択します。
5. **Job version** ドロップダウンからバージョンを選択します。
6. （オプション）新しいジョブのハイパーパラメーターの上書きを提供します。
7. **Destination project** ドロップダウンからプロジェクトを選択します。
8. キューにジョブをエンキューします。
9. **Next step** をクリックします。
10. **Automation name** フィールドにWebhook自動化の名前を入力します。
11. （オプション）Webhook の説明を提供します。
12. **Create automation** ボタンをクリックします。

モデルCIで W&B Launch の自動化を作成する方法について、エンドツーエンドの例はこの[レポート](https://wandb.ai/examples/wandb_automations/reports/Model-CI-with-W-B-Automations--Vmlldzo0NDY5OTIx)をご覧ください。

## 自動化を見る

W&B アプリケーション UI から登録されたモデルに関連する自動化を確認します。

1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model) の Model Registry アプリに移動します。
2. 登録されたモデルを選択します。
3. ページの下部にある **Automations** セクションまでスクロールします。

Automations セクション内では、選択したモデルに対して作成された自動化の以下のプロパティを確認できます：

- **Trigger type**: 設定されたトリガーのタイプ。
- **Action type**: 自動化をトリガーするアクションタイプ。利用可能なオプションは Webhooks と Launch です。
- **Action name**: 自動化を作成する際に提供されたアクション名。
- **Queue**: ジョブがエンキューされたキューの名前。Webhook アクションタイプを選択した場合、このフィールドは空のままです。

## 自動化を削除する

モデルに関連付けられた自動化を削除します。アクションが完了する前にその自動化を削除した場合でも、進行中のアクションには影響しません。

1. [https://wandb.ai/registry/model](https://wandb.ai/registry/model) の Model Registry アプリに移動します。
2. 登録されたモデルをクリックします。
3. ページの下部にある **Automations** セクションまでスクロールします。
4. 自動化の名前の横にカーソルを合わせ、垂直ドット（三点）メニューをクリックします。
5. **Delete** を選択します。