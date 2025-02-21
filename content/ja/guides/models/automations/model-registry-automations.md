---
title: Model registry automations
description: Automation を使用して、モデル の CI (自動化された モデル の 評価 パイプライン ) および モデル の デプロイメント
  を行います。
menu:
  default:
    identifier: ja-guides-models-automations-model-registry-automations
    parent: automations
url: guides/model_registry/model-registry-automations
---

自動化を作成して、自動モデルのテストやデプロイメントなどのワークフローのステップをトリガーします。自動化を作成するには、[イベントタイプ]({{< relref path="#event-types" lang="ja" >}})に基づいて発生させたいアクションを定義します。

たとえば、新しいバージョンの Registered Model を追加したときに、モデルを自動的に GitHub にデプロイするトリガーを作成できます。

{{% alert %}}
オートメーションに関するコンパニオンチュートリアルをお探しですか？
1. [こちらの](https://wandb.ai/wandb/wandb-model-cicd/reports/Model-CI-CD-with-W-B--Vmlldzo0OTcwNDQw) チュートリアルでは、モデルの評価とデプロイメントのために Github Action をトリガーする自動化を設定する方法を紹介します。
2. [こちらの](https://youtube.com/playlist?list=PLD80i8An1OEGECFPgY-HPCNjXgGu-qGO6&feature=shared) ビデオシリーズでは、webhook の基本と、W&B での設定方法を紹介します。
3. [こちらの](https://www.youtube.com/watch?v=s5CMj_w3DaQ) デモでは、モデルを Sagemaker Endpoint にデプロイするための自動化の設定方法を詳しく説明します。
{{% /alert %}}

## イベントタイプ
*イベント* とは、W&B エコシステムで発生する変更のことです。Model Registry は、次の 2 つのイベントタイプをサポートしています。

- 新しいモデル候補をテストするには、**新しい Artifacts を Registered Model にリンクする** を使用します。
- Registered Model のバージョンの `deploy` などのワークフローの特別なステップを表すエイリアスを指定するには、**Registered Model のバージョンに新しいエイリアスを追加する** を使用します。新しいモデルバージョンにそのエイリアスが適用されるたびに実行されます。

[モデルバージョンのリンク]({{< relref path="/guides/models/registry/link_version.md" lang="ja" >}}) および [カスタムエイリアスの作成]({{< relref path="/guides/core/artifacts/create-a-custom-alias.md" lang="ja" >}}) を参照してください。

## Webhook オートメーションの作成
W&B App UI を使用して、アクションに基づいて webhook を自動化します。これを行うには、まず webhook を確立し、次に webhook オートメーションを構成します。

{{% alert %}}
webhook のエンドポイントには、完全修飾ドメイン名が必要です。W&B は、IP アドレスまたは `localhost` などのホスト名によるエンドポイントへの接続をサポートしていません。この制限は、サーバーサイドのリクエストフォージェリ（SSRF）攻撃やその他の関連する脅威ベクトルから保護するのに役立ちます。
{{% /alert %}}

### 認証または認可のためのシークレットの追加
シークレットは、認証情報、 APIキー 、パスワード、トークンなどのプライベート文字列を難読化できる Team レベルの変数です。W&B では、プレーンテキストの内容を保護したい文字列を保存するためにシークレットを使用することを推奨しています。

webhook でシークレットを使用するには、まずそのシークレットを Team のシークレットマネージャーに追加する必要があります。

{{% alert %}}
* シークレットを作成、編集、または削除できるのは、W&B の管理者のみです。
* HTTP POST リクエストを送信する外部サーバーがシークレットを使用しない場合は、このセクションをスキップしてください。
* Azure、GCP、または AWS デプロイメントで [W&B Server]({{< relref path="/guides/hosting/" lang="ja" >}}) を使用する場合にも、シークレットを使用できます。別のデプロイメントタイプを使用している場合に、W&B でシークレットを使用する方法については、W&B アカウント Team にお問い合わせください。
{{% /alert %}}

webhook オートメーションを使用する場合、W&B が作成を推奨するシークレットには、次の 2 種類があります。

* **アクセストークン**：送信者を認証して、webhook リクエストのセキュリティを強化します。
* **シークレット**：ペイロードから送信されるデータの信頼性と整合性を確保します。

webhook を作成するには、次の手順に従います。

1. W&B App UI に移動します。
2. **Team 設定** をクリックします。
3. **Team シークレット** セクションが見つかるまでページを下にスクロールします。
4. **新しいシークレット** ボタンをクリックします。
5. モーダルが表示されます。**シークレット名** フィールドにシークレットの名前を入力します。
6. **シークレット** フィールドにシークレットを追加します。
7. （オプション）webhook が webhook を認証するために追加のシークレットキーまたはトークンを必要とする場合は、手順 5 と 6 を繰り返して別のシークレット（アクセストークンなど）を作成します。

webhook を構成する際に、webhook オートメーションに使用するシークレットを指定します。詳細については、[webhook の構成]({{< relref path="#configure-a-webhook" lang="ja" >}}) セクションを参照してください。

{{% alert %}}
シークレットを作成すると、`$` を使用して W&B ワークフローでそのシークレットにアクセスできます。
{{% /alert %}}

{{% alert color="secondary" %}}
W&B Server でシークレットを使用する場合は、セキュリティニーズを満たすセキュリティ対策を構成する責任があります。

W&B では、AWS、GCP、または Azure が提供するクラウドシークレットマネージャーの W&B インスタンスにシークレットを保存することを強く推奨しています。AWS、GCP、および Azure が提供するシークレットマネージャーは、高度なセキュリティ機能で構成されています。

Kubernetes クラスターをシークレットストアのバックエンドとして使用することはお勧めしません。クラウドシークレットマネージャー（AWS、GCP、または Azure）の W&B インスタンスを使用できない場合にのみ Kubernetes クラスターを検討し、クラスターを使用した場合に発生する可能性のあるセキュリティの脆弱性を防止する方法を理解している必要があります。
{{% /alert %}}

### Webhook の構成
webhook を使用する前に、まず W&B App UI でその webhook を構成します。

{{% alert %}}
* W&B Team の webhook を構成できるのは、W&B 管理者のみです。
* webhook が webhook を認証するために追加のシークレットキーまたはトークンを必要とする場合は、事前に [1 つ以上のシークレットを作成]({{< relref path="#add-a-secret-for-authentication-or-authorization" lang="ja" >}}) していることを確認してください。
{{% /alert %}}

1. W&B App UI に移動します。
2. **Team 設定** をクリックします。
4. **Webhook** セクションが見つかるまでページを下にスクロールします。
5. **新しい webhook** ボタンをクリックします。
6. **名前** フィールドに webhook の名前を入力します。
7. **URL** フィールドに webhook のエンドポイント URL を入力します。
8. （オプション）**シークレット** ドロップダウンメニューから、webhook ペイロードの認証に使用するシークレットを選択します。
9. （オプション）**アクセストークン** ドロップダウンメニューから、送信者を認証するために使用するアクセストークンを選択します。
9. （オプション）**アクセストークン** ドロップダウンメニューから、webhook の認証に必要な追加のシークレットキーまたはトークン（アクセストークンなど）を選択します。

{{% alert %}}
POST リクエストでシークレットとアクセストークンがどこに指定されているかを確認するには、[Webhook のトラブルシューティング]({{< relref path="#troubleshoot-your-webhook" lang="ja" >}}) セクションを参照してください。
{{% /alert %}}

### Webhook の追加
webhook を構成し、（オプションで）シークレットを設定したら、Model Registry アプリ（[https://wandb.ai/registry/model](https://wandb.ai/registry/model)）に移動します。

1. **イベントタイプ** ドロップダウンから、[イベントタイプ]({{< relref path="#event-types" lang="ja" >}})を選択します。
{{< img src="/images/models/webhook_select_event.png" alt="" >}}
2. （オプション）**新しいバージョンが Registered Model に追加されました** イベントを選択した場合は、**Registered Model** ドロップダウンから Registered Model の名前を入力します。
{{< img src="/images/models/webhook_new_version_reg_model.png" alt="" >}}
3. **アクションタイプ** ドロップダウンから **Webhook** を選択します。
4. **次のステップ** ボタンをクリックします。
5. **Webhook** ドロップダウンから webhook を選択します。
{{< img src="/images/models/webhooks_select_from_dropdown.png" alt="" >}}
6. （オプション）JSON 式エディターでペイロードを入力します。一般的なユースケースの例については、[ペイロードの例]({{< relref path="#example-payloads" lang="ja" >}}) セクションを参照してください。
7. **次のステップ** をクリックします。
8. **オートメーション名** フィールドに webhook オートメーションの名前を入力します。
{{< img src="/images/models/webhook_name_automation.png" alt="" >}}
9. （オプション）webhook の説明を入力します。
10. **オートメーションの作成** ボタンをクリックします。

### ペイロードの例

次のタブは、一般的なユースケースに基づいたペイロードの例を示しています。例では、ペイロードパラメータの条件オブジェクトを参照するために、次のキーを参照しています。
* `${event_type}` アクションをトリガーしたイベントのタイプを参照します。
* `${event_author}` アクションをトリガーした User を参照します。
* `${artifact_version}` アクションをトリガーした特定の Artifacts バージョンを参照します。Artifacts インスタンスとして渡されます。
* `${artifact_version_string}` アクションをトリガーした特定の Artifacts バージョンを参照します。文字列として渡されます。
* `${artifact_collection_name}` Artifacts バージョンがリンクされている Artifacts Collection の名前を参照します。
* `${project_name}` アクションをトリガーしたミューテーションを所有する Project の名前を参照します。
* `${entity_name}` アクションをトリガーしたミューテーションを所有する Entity の名前を参照します。

{{< tabpane text=true >}}
{{% tab header="GitHub リポジトリディスパッチ" value="github" %}}

{{% alert %}}
アクセストークンに、GHA ワークフローをトリガーするために必要な権限セットがあることを確認してください。詳細については、[GitHub のドキュメントを参照してください](https://docs.github.com/en/rest/repos/repos?#create-a-repository-dispatch-event)。
{{% /alert %}}

W&B からリポジトリディスパッチを送信して、GitHub アクションをトリガーします。たとえば、`on` キーのトリガーとしてリポジトリディスパッチを受け入れるワークフローがあるとします。

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

  レンダリングされたテンプレート文字列の内容と位置は、オートメーションが構成されているイベントまたはモデルバージョンによって異なります。`${event_type}` は、`LINK_ARTIFACT` または `ADD_ARTIFACT_ALIAS` としてレンダリングされます。以下にマッピングの例を示します。

  ```json
  ${event_type} --> "LINK_ARTIFACT" または "ADD_ARTIFACT_ALIAS"
  ${event_author} --> "<wandb-user>"
  ${artifact_version} --> "wandb-artifact://_id/QXJ0aWZhY3Q6NTE3ODg5ODg3""
  ${artifact_version_string} --> "<entity>/model-registry/<registered_model_name>:<alias>"
  ${artifact_collection_name} --> "<registered_model_name>"
  ${project_name} --> "model-registry"
  ${entity_name} --> "<entity>"
  ```

  テンプレート文字列を使用して、W&B から GitHub Actions やその他のツールにコンテキストを動的に渡します。これらのツールが Python スクリプトを呼び出すことができる場合は、[W&B API]({{< relref path="/guides/core/artifacts/download-and-use-an-artifact.md" lang="ja" >}}) を使用して Registered Model Artifacts を使用できます。

  リポジトリディスパッチの詳細については、[GitHub Marketplace の公式ドキュメント](https://github.com/marketplace/actions/repository-dispatch)を参照してください。

  モデルの評価とデプロイメントのための自動化を作成する手順については、[モデル評価のためのWebhook 自動化](https://www.youtube.com/watch?v=7j-Mtbo-E74&ab_channel=Weights%26Biases) および [モデルデプロイメントのためのWebhook 自動化](https://www.youtube.com/watch?v=g5UiAFjM2nA&ab_channel=Weights%26Biases) のビデオをご覧ください。

  Model CI に Github Actions webhook オートメーションを使用する方法を示す W&B [レポート](https://wandb.ai/wandb/wandb-model-cicd/reports/Model-CI-CD-with-W-B--Vmlldzo0OTcwNDQw) を確認してください。Modal Labs webhook でモデル CI を作成する方法については、[GitHub リポジトリ](https://github.com/hamelsmu/wandb-modal-webhook) を確認してください。

{{% /tab %}}

{{% tab header="Microsoft Teams 通知" value="microsoft"%}}

  構成することにより、Teams Channel の webhook URL を取得するために ‘Incoming Webhook' を構成します。以下はペイロードの例です。
  
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

  テンプレート文字列を使用すると、実行時に W&B データをペイロードに挿入できます（上記の Teams の例を参照）。

{{% /tab %}}

{{% tab header="Slack 通知" value="slack"%}}

  [Slack API ドキュメント](https://api.slack.com/messaging/webhooks) に記載されている手順に従って、Slack アプリを設定し、受信 webhook インテグレーションを追加します。`Bot User OAuth Toke`n で指定されたシークレットが W&B webhook のアクセストークンであることを確認してください。
  
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

### Webhook のトラブルシューティング

W&B App UI を使用してインタラクティブに、または Bash スクリプトを使用してプログラムで webhook のトラブルシューティングを行います。新しい webhook を作成するとき、または既存の webhook を編集するときに、webhook のトラブルシューティングを行うことができます。

{{< tabpane text=true >}}
{{% tab header="W&B App UI" value="app" %}}

  W&B App UI を使用して webhook をインタラクティブにテストします。

  1. W&B Team 設定ページに移動します。
  2. **Webhook** セクションまでスクロールします。
  3. webhook の名前の横にある水平方向の 3 つのドキュメント（ミートボールアイコン）をクリックします。
  4. **テスト** を選択します。
  5. 表示される UI パネルから、表示されるフィールドに POST リクエストを貼り付けます。
     {{< img src="/images/models/webhook_ui.png" >}}
  6. **Webhook のテスト** をクリックします。

  W&B App UI 内で、W&B はエンドポイントによって行われた応答を投稿します。

  {{< img src="/images/models/webhook_ui_testing.gif" alt="" >}}

  実際の例については、[Weights & Biases での Webhook のテスト](https://www.youtube.com/watch?v=bl44fDpMGJw&ab_channel=Weights%26Biases) のビデオをご覧ください。

{{% /tab %}}

{{% tab header="Bash スクリプト" value="bash" %}}

  次の Bash スクリプトは、W&B がトリガーされたときに webhook オートメーションに送信する POST リクエストと同様の POST リクエストを生成します。

  以下のコードをコピーしてシェルスクリプトに貼り付け、webhook のトラブルシューティングを行います。以下について独自の値 ( Value ) を指定します。

  * `ACCESS_TOKEN`
  * `SECRET`
  * `PAYLOAD`
  * `API_ENDPOINT`

  ```sh { title = "webhook_test.sh" }
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
  # セキュリティのため、Wandb は、HMAC with SHA-256 アルゴリズムを使用して、
  # ペイロードと webhook に関連付けられた共有シークレットキーから計算されたヘッダーに X-Wandb-Signature を含めます。
  SIGNATURE=$(echo -n "$PAYLOAD" | openssl dgst -sha256 -hmac "$SECRET" -binary | base64)

  # Make the cURL request
  curl -X POST \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $ACCESS_TOKEN" \
    -H "X-Wandb-Signature: $SIGNATURE" \
    -d "$PAYLOAD" API_ENDPOINT
  ```

{{% /tab %}}
{{< /tabpane >}}

## オートメーションの表示

W&B App UI から Registered Model に関連付けられたオートメーションを表示します。

1. Model Registry アプリ（[https://wandb.ai/registry/model](https://wandb.ai/registry/model)）に移動します。
2. Registered Model を選択します。
3. ページの最下部までスクロールして、**オートメーション** セクションを表示します。

オートメーションセクション内には、選択したモデルに対して作成されたオートメーションの次のプロパティがあります。

- **トリガータイプ**：構成されたトリガーのタイプ。
- **アクションタイプ**：オートメーションをトリガーするアクションタイプ。
- **アクション名**：オートメーションを作成したときに指定したアクション名。
- **キュー**：ジョブがエンキューされたキューの名前。webhook アクションタイプを選択した場合、このフィールドは空のままになります。

## オートメーションの削除
モデルに関連付けられたオートメーションを削除します。アクションが完了する前にそのオートメーションを削除した場合、進行中のアクションは影響を受けません。

1. Model Registry アプリ（[https://wandb.ai/registry/model](https://wandb.ai/registry/model)）に移動します。
2. Registered Model をクリックします。
3. ページの最下部までスクロールして、**オートメーション** セクションを表示します。
4. オートメーションの名前の横にマウスを置き、ケバブ（縦に 3 つのドット）メニューをクリックします。
5. **削除** を選択します。
