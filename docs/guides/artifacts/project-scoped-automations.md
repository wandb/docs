---
title: Artifact automations
description: プロジェクトのスコープに限定されたアーティファクトオートメーションを使用して、アーティファクトコレクション内でエイリアスやバージョンが作成または変更されたときにアクションをトリガーします。
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# アーティファクトの変更によるCI/CDイベントのトリガー

アーティファクトが変更された際にトリガーされるオートメーションを作成します。アーティファクトのバージョン管理のために下流アクションを自動化したい場合、アーティファクトオートメーションを使用します。オートメーションを作成するには、[イベントタイプ](#event-types)に基づいて発生させたい[アクション](#action-types)を定義します。

アーティファクトの変更からトリガーされるオートメーションの一般的なユースケースとしては次のようなものがあります：

* 新しいバージョンの評価/ホールドアウトデータセットがアップロードされたときに、モデルレジストリの最良のトレーニングモデルを使用して推論を行い、パフォーマンス情報を含むレポートを作成する[ローンチジョブをトリガー](#create-a-launch-automation)します。
* トレーニングデータセットの新しいバージョンが「プロダクション」としてラベル付けされたときに、現在の最良のパフォーマンスモデルの設定を使用して[リトレーニングローンチ](#create-a-launch-automation)ジョブをトリガーします。

:::info
アーティファクトオートメーションはプロジェクトにスコープされています。これは、プロジェクト内のできごとのみがアーティファクトオートメーションをトリガーすることを意味します。

これに対して、Weights & Biasesモデルレジストリで作成されたオートメーションはモデルレジストリのスコープにあります。モデルバージョンに対してイベントが実行されたときにトリガーされます。モデルバージョンのオートメーションを作成する方法については、[Model CI/CDのオートメーション](../model_registry/model-registry-automations.md)ページを参照してください。[モデルレジストリチャプター](../model_registry/intro.md)に含まれる情報も参考にしてください。
:::

## イベントタイプ
*イベント*は、Weights & Biasesエコシステム内で発生する変更を意味します。プロジェクト内のアーティファクトコレクションに対して、次の2種類のイベントタイプを定義できます： **コレクション内のアーティファクトの新しいバージョンが作成される** と **アーティファクトエイリアスが追加される**。

:::tip
**コレクション内のアーティファクトの新しいバージョンが作成される**イベントタイプは、アーティファクトの各バージョンに繰り返しアクションを適用するために使用します。例えば、新しいデータセットアーティファクトバージョンが作成されたときに自動的にトレーニングジョブを開始するオートメーションを作成できます。

**アーティファクトエイリアスが追加される**イベントタイプは、特定のエイリアスがアーティファクトバージョンに適用されたときにアクティブになるオートメーションを作成するために使用します。例えば、「test-set-quality-check」エイリアスがアーティファクトに追加されたときにデータセットで下流プロセッシングをトリガーするアクションをトリガーするオートメーションを作成できます。
:::

## アクションタイプ
アクションは、特定のトリガーに基づいて発生する応答的な変動（内部または外部）です。プロジェクト内のアーティファクトコレクションに関するイベントに対応するアクションには、ウェブフックと[W&B Launch Jobs](../launch/intro.md)の2種類があります。

* ウェブフック: HTTPリクエストを通じてWeights & Biasesから外部のウェブサーバーと通信します。
* W&B Launch Job: [Jobs](../launch/create-launch-job.md) は再利用可能で設定可能な実行テンプレートで、ローカルデスクトップやEKSのKubernetes、Amazon SageMakerなどの外部計算リソースでの新しい [runs](../runs/intro.md) の迅速なローンチを可能にします。

以下のセクションでは、ウェブフックとW&B Launchを使用してオートメーションを作成する方法について説明します。

## ウェブフックオートメーションの作成
W&BアプリUIを使用して、アクションに基づいてウェブフックを自動化します。このためには、まずウェブフックを設定し、次にウェブフックオートメーションを設定します。

:::info
エンドポイントのアドレスレコード(Aレコード)を指定してください。Weights & Biasesは、`[0-255].[0-255].[0-255].[0.255]`のような直接的にIPアドレスで公開されているエンドポイントや`localhost`で公開されているエンドポイントへの接続をサポートしていません。この制限は、サーバーサイドリクエストフォージェリ(SSRF)攻撃やその他の関連する脅威ベクトルから保護するためです。
:::

### 認証または認可のためのシークレットを追加
シークレットは、資格情報、APIキー、パスワード、トークンなどの秘密の文字列を隠すことができるチームレベルの変数です。Weights & Biasesは、平文の内容を保護したい文字列を保存するためにシークレットを使用することを推奨しています。

ウェブフックでシークレットを使用するには、まずそのシークレットをチームのシークレットマネージャーに追加する必要があります。

:::info
* W&Bの管理者のみがシークレットを作成、編集、削除できます。
* 送信先の外部サーバーがシークレットを使用しない場合、このセクションはスキップしてください。
* また、Azure、GCP、AWSデプロイメントで [W&B Server](../hosting/intro.md) を使用する場合にもシークレットが利用可能です。異なるデプロイメントタイプを使用する場合には、W&Bアカウントチームと連絡を取り、シークレットの使用方法についてご相談ください。
:::

ウェブフックオートメーションを使用する場合には、Weights & Biasesで次の2種類のシークレットを作成することをお勧めします：

* **アクセストークン**: ウェブフックリクエストを保護するために送信者を認可します。
* **シークレット**: ペイロードから送信されたデータの信憑性と整合性を確保します。

以下の手順に従ってウェブフックを作成します：

1. W&BアプリUIに移動します。
2. **Team Settings** をクリックします。
3. ページを下にスクロールし、**Team secrets** セクションを見つけます。
4. **New secret** ボタンをクリックします。
5. モーダルが表示されます。**Secret name** フィールドにシークレットの名前を入力します。
6. **Secret** フィールドにシークレットを入力します。
7. (オプション) ウェブフックに追加のシークレットキーやトークンが必要な場合、手順5と6を繰り返して別のシークレット（例えばアクセストークン）を作成します。

ウェブフックを設定する際に、ウェブフックオートメーションで使用するシークレットを指定します。詳細は[ウェブフックの設定](#configure-a-webhook)セクションを参照してください。

:::tip
シークレットを作成した後、W&Bワークフロー内でそのシークレットに `$` を使ってアクセスできます。
:::

### ウェブフックの設定
ウェブフックを使用する前に、まずW&BアプリUIでウェブフックを設定する必要があります。

:::info
* W&Bの管理者のみがW&Bチームのためにウェブフックを設定できます。
* ウェブフックが認証に追加のシークレットキーやトークンを必要とする場合、すでに [シークレットを作成している](#add-a-secret-for-authentication-or-authorization) ことを確認してください。
:::

1. W&BアプリUIに移動します。
2. **Team Settings** をクリックします。
3. ページを下にスクロールし、**Webhooks** セクションを見つけます。
4. **New webhook** ボタンをクリックします。
5. **Name** フィールドにウェブフックの名前を入力します。
6. **URL** フィールドにウェブフックのエンドポイントURLを入力します。
7. (オプション) **Secret** ドロップダウンメニューから、ウェブフックペイロードを認証するために使用するシークレットを選択します。
8. (オプション) **Access token** ドロップダウンメニューから、送信者を認可するために使用するアクセストークンを選択します。
8. (オプション) **Access token** ドロップダウンメニューから、ウェブフックを認証するために必要な追加のシークレットキーやトークン（例えばアクセストークン）を選択します。

:::note
POSTリクエスト内でシークレットとアクセストークンが指定された場所については、[ウェブフックのトラブルシューティング](#troubleshoot-your-webhook)セクションを参照してください。
:::

### ウェブフックの追加
ウェブフックを設定し（オプションでシークレットも設定）したら、プロジェクトワークスペースに移動します。左サイドバーの **Automations** タブをクリックします。

1. **Event type** ドロップダウンから[イベントタイプ](#event-types)を選択します。
![](/images/artifacts/artifact_webhook_select_event.png)
2. **コレクション内のアーティファクトの新しいバージョンが作成される**イベントを選択した場合、**Artifact collection** ドロップダウンからオートメーションが反応すべきアーティファクトコレクションの名前を提供します。
![](/images/artifacts/webhook_new_version_artifact.png)
3. **Action type** ドロップダウンから **Webhooks** を選択します。
4. **Next step** ボタンをクリックします。
5. **Webhook** ドロップダウンからウェブフックを選択します。
![](/images/artifacts/artifacts_webhooks_select_from_dropdown.png)
6. (オプション) JSON式エディタにペイロードを提供します。一般的なユースケースの例については[例のペイロード](#example-payloads)セクションを参照してください。
7. **Next step** をクリックします。
8. **Automation name** フィールドにウェブフックオートメーションの名前を入力します。
![](/images/artifacts/artifacts_webhook_name_automation.png)
9. (オプション)ウェブフックに説明を提供します。
10. **Create automation** ボタンをクリックします。

### 例のペイロード

以下のタブは、一般的なユースケースに基づいた例のペイロードを示しています。例の中では、ペイロードパラメータ内の条件オブジェクトを参照するために以下のキーが使用されています：
* `${event_type}` は、アクションをトリガーしたイベントの種類を指します。
* `${event_author}` は、アクションをトリガーしたユーザーを指します。
* `${artifact_version}` は、アクションをトリガーした特定のアーティファクトバージョンを指します。アーティファクトインスタンスとして渡されます。
* `${artifact_version_string}` は、アクションをトリガーした特定のアーティファクトバージョンを指します。文字列として渡されます。
* `${artifact_collection_name}` は、アーティファクトバージョンがリンクされているアーティファクトコレクションの名前を指します。
* `${project_name}` は、アクションをトリガーした変更を所有するプロジェクトの名前を指します。
* `${entity_name}` は、アクションをトリガーした変更を所有するエンティティの名前を指します。

<Tabs
  defaultValue="github"
  values={[
    {label: 'GitHub repository dispatch', value: 'github'},
    {label: 'Microsoft Teams notification', value: 'microsoft'},
    {label: 'Slack notifications', value: 'slack'},
  ]}>
  <TabItem value="github">

:::info
アクセス トークンに、GitHub Actions ワークフローをトリガーするために必要な権限セットが設定されていることを確認してください。詳細については、[GitHub Docs](https://docs.github.com/en/rest/repos/repos?#create-a-repository-dispatch-event) を参照してください。
:::

  W&Bからリポジトリディスパッチを送信してGitHubアクションをトリガーします。例えば、`on`キーのトリガーとしてリポジトリディスパッチを受け入れるワークフローがあるとしましょう：

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
ウェブフックペイロード内の `event_type` キーは、GitHubワークフローYAMLファイルの `types` フィールドと一致する必要があります。
:::

  レンダリングされたテンプレート文字列の内容と位置は、設定されたイベントやモデルバージョンによって異なります。`${event_type}` は "LINK_ARTIFACT" または "ADD_ARTIFACT_ALIAS" として表示されます。以下は例のマッピングです：

  ```json
  ${event_type} --> "LINK_ARTIFACT" または "ADD_ARTIFACT_ALIAS"
  ${event_author} --> "<wandb-user>"
  ${artifact_version} --> "wandb-artifact://_id/QXJ0aWZhY3Q6NTE3ODg5ODg3"
  ${artifact_version_string} --> "<entity>/<project_name>/<artifact_name>:<alias>"
  ${artifact_collection_name} --> "<artifact_collection_name>"
  ${project_name} --> "<project_name>"
  ${entity_name} --> "<entity>"
  ```

  テンプレート文字列を使用して、W&BからGitHub Actionsや他のツールへのコンテキストを動的に渡します。これらのツールがPythonスクリプトを呼び出せる場合、[W&B API](../artifacts/download-and-use-an-artifact.md)を介してW&Bアーティファクトを利用できます。

  リポジトリディスパッチの詳細については、[GitHub Marketplaceの公式ドキュメント](https://github.com/marketplace/actions/repository-dispatch) を参照してください。

  </TabItem>
  <TabItem value="microsoft">

  「受信ウェブフック」を設定して、TeamsチャンネルのためにウェブフックURLを取得します。以下は例のペイロードです：

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
  上記のTeamsの例のように、テンプレート文字列を使用して実行時にW&Bデータをペイロードに挿入することができます。

  </TabItem>
  <TabItem value="slack">

  Slackアプリを設定し、[Slack APIドキュメント](https://api.slack.com/messaging/webhooks)に記載された指示に従って受信ウェブフック統合を追加します。W&Bウェブフックのアクセストークンとして「Bot User OAuth Token」が指定されていることを確認してください。

  以下は例のペイロードです：

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

### ウェブフックのトラブルシューティング

W&BアプリUIを使ってインタラクティブにウェブフックをトラブルシュートするか、Bashスクリプトを使ってプログラム的に行います。新しいウェブフックを作成する際や既存のウェブフックを編集する際に、ウェブフックをトラブルシュートできます。

<Tabs
  defaultValue="app"
  values={[
    {label: 'W&B App UI', value: 'app'},
    {label: 'Bash script', value: 'bash'},
  ]}>
  <TabItem value="app">

W&BアプリUIを使ってインタラクティブにウェブフックをテストします。

1. W&Bチーム設定ページに移動します。
2. **Webhooks** セクションまでスクロールします。
3. ウェブフックの名前の横にある横三つのドット（ミートボールアイコン）をクリックします。
4. **Test** を選択します。
5. 表示されるUIパネルから、POSTリクエストを貼り付けます。
![](/images/models/webhook_ui.png)
6. **Test webhook** ボタンをクリックします。

W&BアプリUI内で、エンドポイントによるレスポンスが表示されます。

![](/images/models/webhook_ui_testing.gif)

YouTubeの[Testing Webhooks in Weights & Biases](https://www.youtube.com/watch?v=bl44fDpMGJw&ab_channel=Weights%26Biases)ビデオで、実際の例を確認できます。

  </TabItem>
  <TabItem value="bash">

以下のbashスクリプトは、ウェブフックオートメーションがトリガーされた際にW&Bが送信するPOSTリクエストに似たリクエストを生成します。

コードを以下にコピーし、シェルスクリプトに貼り付けてウェブフックをトラブルシュートします。以下の値を指定してください：

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

## ローンチオートメーションの作成
W&Bジョブを自動的に開始します。

:::info
このセクションでは、すでにジョブを作成し、キューを設定し、エージェントがポーリング中であることを前提としています。詳細については、[W&B Launch docs](../launch/intro.md)を参照してください。
:::

1. **Event type** ドロップダウンからイベントタイプを選択します。サポートされているイベントの詳細は[イベントタイプ](#event-types)セクションを参照してください。
2. (オプション) **コレクション内のアーティファクトの新しいバージョンが作成される**イベントを選択した場合、**Artifact collection** ドロップダウンからアーティファクトコレクションの名前を指定します。
3. **Action type** ドロップダウンから **Jobs** を選択します。
4. **Next step** をクリックします。
4. **Job** ドロップダウンからW&B Launchジョブを選択します。
5. **Job version** ドロップダウンからバージョンを選択します。
6. (オプション) 新しいジョブのハイパーパラメータを上書きします。
7. **Destination project**ドロップダウンからプロジェクトを選択します。
8. キューにジョブをエンキューします。
9. **Next step**をクリックします。
10. **Automation name** フィールドにウェブフックオートメーションの名前を入力します。
11. (オプション) ウェブフックの説明を提供します。
12. **Create automation** ボタンをクリックします。

## オートメーションの表示

W&BアプリUIからアーティファクトに関連するオートメーションを表示します。

1. W&Bアプリのプロジェクトワークスペースに移動します。
2. 左側のサイドバーで **Automations** タブをクリックします。

![](/images/artifacts/automations_sidebar.gif)

Automationsセクションでは、プロジェクトで作成された各オートメーションの以下のプロパティを確認できます：

- **Trigger type**: 設定されたトリガーのタイプ。
- **Action type**: オートメーションをトリガーするアクションタイプ。利用可能なオプションはWebhooksとLaunchです。
- **Action name**: オートメーションを作成するときに提供したアクション名。
- **Queue**: ジョブがエンキューされたキューの名前。このフィールドは、ウェブフックアクションタイプを選択した場合には空のままです。

## オートメーションの削除
アーティファクトに関連付けられたオートメーションを削除します。アクションが完了する前にオートメーションを削除した場合、進行中のアクションには影響しません。

