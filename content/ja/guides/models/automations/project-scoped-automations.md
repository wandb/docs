---
title: Trigger CI/CD events when artifact changes
description: プロジェクト スコープのアーティファクト自動化をプロジェクトで使用して、アーティファクトコレクション内のエイリアスまたはバージョンが作成または変更されたときにアクションをトリガーします。
menu:
  default:
    identifier: ja-guides-models-automations-project-scoped-automations
    parent: automations
url: guides/artifacts/project-scoped-automations
---

Artifacts が変更されたときにトリガーされるオートメーションを作成します。Artifacts の バージョン管理 のためのダウンストリームアクションを自動化する場合は、Artifacts オートメーションを使用します。オートメーションを作成するには、[イベントタイプ]({{< relref path="#event-types" lang="ja" >}}) に基づいて発生させたいアクションを定義します。

{{% alert %}}
Artifacts オートメーションは、プロジェクトのスコープに設定されています。つまり、プロジェクト内のイベントのみが Artifacts オートメーションをトリガーします。

これは、W&B Model Registry で作成されたオートメーションとは対照的です。Model Registry で作成されたオートメーションは、Model Registry のスコープ内にあります。これらは、[Model Registry]({{< relref path="/guides/models/registry/model_registry/" lang="ja" >}}) にリンクされたモデルバージョンでイベントが実行されたときにトリガーされます。モデルバージョンのオートメーションを作成する方法については、[Model CI/CD のためのオートメーション]({{< relref path="/guides/models/automations/model-registry-automations.md" lang="ja" >}}) ページの [Model Registry chapter]({{< relref path="/guides/models/registry/model_registry/" lang="ja" >}}) を参照してください。
{{% /alert %}}

## イベントタイプ
*イベント* とは、W&B エコシステム で発生する変更のことです。プロジェクトの Artifacts コレクション に対して、**Artifacts の新しい バージョン がコレクションに作成された** と **Artifacts の エイリアス が追加された** という 2 種類のイベントタイプを定義できます。

{{% alert %}}
Artifacts の各 バージョン に定期的なアクションを適用するには、**Artifacts の新しい バージョン がコレクションに作成された** イベントタイプを使用します。たとえば、新しいデータセット Artifacts バージョン が作成されたときに、自動的にトレーニング ジョブを開始するオートメーションを作成できます。

特定の エイリアス が Artifacts バージョン に適用されたときにアクティブ化されるオートメーションを作成するには、**Artifacts の エイリアス が追加された** イベントタイプを使用します。たとえば、誰かが「test-set-quality-check」 エイリアス を Artifacts に追加したときにアクションをトリガーし、そのデータセット でダウンストリーム処理をトリガーするオートメーションを作成できます。
{{% /alert %}}

## Webhook オートメーションを作成する
W&B App UI でのアクションに基づいて Webhook を自動化します。これを行うには、まず Webhook を確立し、次に Webhook オートメーションを構成します。

{{% alert %}}
Address レコード (A レコード) を持つ Webhook のエンドポイントを指定します。W&B は、`[0-255].[0-255].[0-255].[0.255]` などの IP アドレスで直接公開されているエンドポイント、または `localhost` として公開されているエンドポイントへの接続をサポートしていません。この制限は、サーバーサイド リクエスト フォージェリ (SSRF) 攻撃やその他の関連する脅威ベクトルから保護するのに役立ちます。
{{% /alert %}}

### 認証または承認のための シークレット を追加する
シークレット は、認証情報、 APIキー 、パスワード、トークンなどのプライベート文字列を難読化できる Team レベルの変数です。W&B では、プレーンテキストコンテンツを保護したい文字列を保存するために シークレット を使用することをお勧めします。

Webhook で シークレット を使用するには、まずその シークレット を Team の シークレット マネージャーに追加する必要があります。

{{% alert %}}
* W&B Admin のみ シークレット を作成、編集、または削除できます。
* HTTP POST リクエスト を送信する外部 サーバー が シークレット を使用しない場合は、このセクションをスキップしてください。
* Azure、GCP、または AWS デプロイメントで [W&B サーバー]({{< relref path="/guides/hosting/" lang="ja" >}}) を使用する場合も、シークレット を利用できます。別のデプロイメントタイプを使用する場合は、W&B アカウント Team に連絡して、W&B で シークレット を使用する方法について話し合ってください。
{{% /alert %}}

Webhook オートメーションを使用する場合、W&B が作成を推奨する シークレット には、次の 2 種類があります。

* **アクセス トークン**: 送信者を承認して、Webhook リクエスト のセキュリティを強化します
* **シークレット**: ペイロード から送信されるデータの信頼性と整合性を確保します

Webhook を作成するには、次の手順に従います。

1. W&B App UI に移動します。
2. [**Team 設定**] をクリックします。
3. [**Team シークレット**] セクションが見つかるまで、ページを下にスクロールします。
4. [**新しい シークレット**] ボタンをクリックします。
5. モーダルが表示されます。[**シークレット 名**] フィールドに シークレット の名前を入力します。
6. [**シークレット**] フィールドに シークレット を追加します。
7. (オプション) Webhook の認証に別の シークレット キーまたはトークンが必要な場合は、手順 5 と 6 を繰り返して別の シークレット (アクセス トークンなど) を作成します。

Webhook を構成するときに、Webhook オートメーションに使用する シークレット を指定します。詳細については、[Webhook の構成]({{< relref path="#configure-a-webhook" lang="ja" >}}) セクションを参照してください。

{{% alert %}}
シークレット を作成すると、W&B ワークフロー で `$` を使用してその シークレット にアクセスできます。
{{% /alert %}}

### Webhook を構成する
Webhook を使用する前に、まず W&B App UI でその Webhook を構成する必要があります。

{{% alert %}}
* W&B Admin のみ W&B Team の Webhook を構成できます。
* Webhook の認証に別の シークレット キーまたはトークンが必要な場合は、[1 つ以上の シークレット をすでに作成している]({{< relref path="#add-a-secret-for-authentication-or-authorization" lang="ja" >}}) ことを確認してください。
{{% /alert %}}

1. W&B App UI に移動します。
2. [**Team 設定**] をクリックします。
4. [**Webhook**] セクションが見つかるまで、ページを下にスクロールします。
5. [**新しい Webhook**] ボタンをクリックします。
6. [**名前**] フィールドに Webhook の名前を入力します。
7. [**URL**] フィールドに Webhook のエンドポイント URL を入力します。
8. (オプション) [**シークレット**] ドロップダウン メニューから、Webhook ペイロード の認証に使用する シークレット を選択します。
9. (オプション) [**アクセス トークン**] ドロップダウン メニューから、送信者の承認に使用する アクセス トークン を選択します。
9. (オプション) [**アクセス トークン**] ドロップダウン メニューから、Webhook の認証に必要な別の シークレット キーまたはトークン (アクセス トークン など) を選択します。

{{% alert %}}
POST リクエスト で シークレット と アクセス トークン が指定されている場所を確認するには、[Webhook のトラブルシューティング]({{< relref path="#troubleshoot-your-webhook" lang="ja" >}}) セクションを参照してください。
{{% /alert %}}

### Webhook を追加する
Webhook を構成し、（オプションで）シークレット を設定したら、プロジェクト ワークスペース に移動します。左側のサイドバーにある [**オートメーション**] タブをクリックします。

1. [**イベントタイプ**] ドロップダウンから、[イベントタイプ]({{< relref path="#event-types" lang="ja" >}}) を選択します。
{{< img src="/images/artifacts/artifact_webhook_select_event.png" alt="" >}}
2. [**Artifacts の新しい バージョン がコレクションに作成された**] イベントを選択した場合は、[**Artifacts コレクション**] ドロップダウンから、オートメーション が応答する Artifacts コレクション の名前を入力します。
{{< img src="/images/artifacts/webhook_new_version_artifact.png" alt="" >}}
3. [**アクションタイプ**] ドロップダウンから [**Webhook**] を選択します。
4. [**次のステップ**] ボタンをクリックします。
5. [**Webhook**] ドロップダウンから Webhook を選択します。
{{< img src="/images/artifacts/artifacts_webhooks_select_from_dropdown.png" alt="" >}}
6. (オプション) JSON 式エディターで ペイロード を指定します。一般的な ユースケース の例については、[ペイロード の例]({{< relref path="#example-payloads" lang="ja" >}}) セクションを参照してください。
7. [**次のステップ**] をクリックします。
8. [**オートメーション 名**] フィールドに Webhook オートメーション の名前を入力します。
{{< img src="/images/artifacts/artifacts_webhook_name_automation.png" alt="" >}}
9. (オプション) Webhook の説明を入力します。
10. [**オートメーション の作成**] ボタンをクリックします。

### ペイロード の例

次のタブは、一般的な ユースケース に基づいた ペイロード の例を示しています。例の中では、次の キー を参照して、 ペイロード パラメータ の条件 オブジェクト を参照しています。
* `${event_type}` アクションをトリガーしたイベントのタイプを参照します。
* `${event_author}` アクションをトリガーした ユーザー を参照します。
* `${artifact_version}` アクションをトリガーした特定の Artifacts バージョン を参照します。Artifacts インスタンスとして渡されます。
* `${artifact_version_string}` アクションをトリガーした特定の Artifacts バージョン を参照します。文字列として渡されます。
* `${artifact_collection_name}` Artifacts バージョン がリンクされている Artifacts コレクション の名前を参照します。
* `${project_name}` アクションをトリガーした変更を所有する Project の名前を参照します。
* `${entity_name}` アクションをトリガーした変更を所有する Entity の名前を参照します。

{{< tabpane text=true >}}

{{% tab header="GitHub repository dispatch" value="github" %}}
{{% alert %}}
アクセス トークン に、GHA ワークフロー をトリガーするために必要な権限セットがあることを確認してください。詳細については、[これらの GitHub ドキュメントを参照してください](https://docs.github.com/en/rest/repos/repos?#create-a-repository-dispatch-event)。
{{% /alert %}}

W&B から リポジトリ ディスパッチ を送信して GitHub アクション をトリガーします。たとえば、`on` キー のトリガーとして リポジトリ ディスパッチ を受け入れる ワークフロー があるとします。

```yaml
on:
  repository_dispatch:
    types: BUILD_AND_DEPLOY
```

リポジトリ の ペイロード は次のようになります。

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
Webhook ペイロード の `event_type` キー は、GitHub ワークフロー YAML ファイル の `types` フィールドと一致する必要があります。
{{% /alert %}}

レンダリングされたテンプレート文字列の内容と位置は、オートメーション が構成されているイベントまたはモデル バージョン によって異なります。`${event_type}` は `LINK_ARTIFACT` または `ADD_ARTIFACT_ALIAS` のいずれかとしてレンダリングされます。マッピングの例を以下に示します。

```json
${event_type} --> "LINK_ARTIFACT" または "ADD_ARTIFACT_ALIAS"
${event_author} --> "<wandb-user>"
${artifact_version} --> "wandb-artifact://_id/QXJ0aWZhY3Q6NTE3ODg5ODg3""
${artifact_version_string} --> "<entity>/<project_name>/<artifact_name>:<alias>"
${artifact_collection_name} --> "<artifact_collection_name>"
${project_name} --> "<project_name>"
${entity_name} --> "<entity>"
```

テンプレート文字列を使用して、W&B から GitHub Actions やその他の ツール にコンテキスト を動的に渡します。これらの ツール が Python スクリプト を呼び出すことができる場合、[W&B API]({{< relref path="/guides/core/artifacts/download-and-use-an-artifact.md" lang="ja" >}}) を通じて W&B Artifacts を使用できます。

リポジトリ ディスパッチ の詳細については、[GitHub Marketplace の公式ドキュメント](https://github.com/marketplace/actions/repository-dispatch) を参照してください。
{{% /tab %}}

{{% tab header="Microsoft Teams notification" value="microsoft"%}}

‘受信 Webhook' を構成して、構成することにより、Team チャンネル の Webhook URL を取得します。以下は ペイロード の例です。
  
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

テンプレート文字列を使用して、実行時に W&B データ を ペイロード に挿入できます (上記の Teams の例に示すように)。

{{% /tab %}}

{{% tab header="Slack notifications" value="slack"%}}

[Slack API ドキュメント](https://api.slack.com/messaging/webhooks) で強調表示されている手順に従って Slack アプリ をセットアップし、受信 Webhook インテグレーション を追加します。`Bot User OAuth Toke` n に指定されている シークレット が W&B Webhook の アクセス トークン であることを確認してください。
  
以下は ペイロード の例です。

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
{{% /tab %}}

{{< /tabpane >}}

### Webhook のトラブルシューティング

W&B App UI を使用してインタラクティブ に、または Bash スクリプト を使用してプログラムで Webhook のトラブルシューティングを行います。新しい Webhook を作成するとき、または既存の Webhook を編集するときに、Webhook のトラブルシューティングを行うことができます。

{{< tabpane text=true >}}
{{% tab header="W&B App UI" value="app" %}}

W&B App UI を使用して Webhook をインタラクティブ にテストします。

1. W&B Team 設定 ページに移動します。
2. [**Webhook**] セクションまでスクロールします。
3. Webhook の名前の横にある水平方向の 3 つのドキュメント (ミートボール アイコン) をクリックします。
4. [**テスト**] を選択します。
5. 表示される UI パネル から、表示されるフィールドに POST リクエスト を貼り付けます。
{{< img src="/images/models/webhook_ui.png" alt="" >}}
6. [**Webhook のテスト**] をクリックします。

W&B App UI 内で、W&B はエンドポイント によって行われた応答を投稿します。

{{< img src="/images/models/webhook_ui_testing.gif" alt="" >}}

実際の例を見るには、[Weights & Biases での Webhook のテスト](https://www.youtube.com/watch?v=bl44fDpMGJw&ab_channel=Weights%26Biases) YouTube ビデオ を参照してください。
{{% /tab %}}

{{% tab header="Bash script" value="bash"%}}

次の bash スクリプト は、トリガーされたときに W&B が Webhook オートメーション に送信する POST リクエスト と同様の POST リクエスト を生成します。

Webhook のトラブルシューティングを行うには、以下のコード をコピーしてシェル スクリプト に貼り付けます。以下に独自の値 (Value) を指定します。

* `ACCESS_TOKEN`
* `SECRET`
* `PAYLOAD`
* `API_ENDPOINT`

{{< prism file="/webhook_test.sh" title="webhook_test.sh">}}{{< /prism >}}

{{% /tab %}}
{{< /tabpane >}}

## オートメーション を表示する

W&B App UI から Artifacts に関連付けられた オートメーション を表示します。

1. W&B App で Project ワークスペース に移動します。
2. 左側のサイドバーにある [**オートメーション**] タブをクリックします。

{{< img src="/images/artifacts/automations_sidebar.gif" alt="" >}}

[オートメーション] セクション内では、Project で作成された各オートメーション の次のプロパティを見つけることができます。

- **トリガー タイプ**: 構成されたトリガー のタイプ。
- **アクション タイプ**: オートメーション をトリガーするアクション タイプ。
- **アクション 名**: オートメーション の作成時に指定したアクション 名。
- **キュー**: ジョブ がエンキュー されたキューの名前。Webhook アクション タイプを選択した場合は、このフィールドは空のままになります。

## オートメーション を削除する
Artifacts に関連付けられた オートメーション を削除します。アクション の完了前に オートメーション を削除しても、進行中のアクション は影響を受けません。

1. W&B App で Project ワークスペース に移動します。
2. 左側のサイドバーにある [**オートメーション**] タブをクリックします。
3. リスト から、表示する オートメーション の名前を選択します。
4. オートメーション の名前の横にマウス カーソル を置き、ケバブ (縦に 3 つのドット) メニューをクリックします。
5. [**削除**] を選択します。
