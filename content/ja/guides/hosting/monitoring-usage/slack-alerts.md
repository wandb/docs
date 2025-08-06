---
title: Slack アラートを設定
menu:
  default:
    identifier: slack-alerts
    parent: monitoring-and-usage
---

W&B サーバーを [Slack](https://slack.com/) と連携しましょう。
{{% alert %}}
[W&B 専用クラウド デプロイメントで Slack アラートを設定するデモ動画](https://www.youtube.com/watch?v=JmvKb-7u-oU)（約6分）をご覧ください。
{{% /alert %}}

## Slack アプリケーションの作成

以下の手順で Slack アプリケーションを作成します。

1. https://api.slack.com/apps にアクセスし、**Create an App** を選択します。

    {{< img src="/images/hosting/create_an_app.png" alt="Create an App ボタン" >}}

2. **App Name** のフィールドにアプリの名前を入力します。
3. 開発するアプリの Slack ワークスペースを選択します。アラートに使用したいワークスペースと同じものを選ぶようにしてください。

    {{< img src="/images/hosting/name_app_workspace.png" alt="アプリの名前とワークスペースの選択" >}}

## Slack アプリケーションの設定

1. 左サイドバーから **OAuth & Permissions** を選択します。

    {{< img src="/images/hosting/add_an_oath.png" alt="OAuth & Permissions メニュー" >}}

2. Scopes セクションで、Bot に **incoming_webhook** scope を追加します。Scope は、開発用ワークスペース内でアプリがアクションを実行する権限を与えます。

    Bot 用 OAuth スコープの詳細は、Slack API ドキュメントの「Understanding OAuth scopes for Bots」チュートリアルをご覧ください。

    {{< img src="/images/hosting/save_urls.png" alt="Bot トークンスコープ" >}}

3. Redirect URL を W&B インストール先の URL に設定します。ローカルシステム設定で指定したホスト URL と同じ URL を使います。インスタンスへの DNS マッピングが複数ある場合は、複数の URL を指定できます。

    {{< img src="/images/hosting/redirect_urls.png" alt="Redirect URLs 設定" >}}

4. **Save URLs** をクリックします。
5. 必要であれば、**Restrict API Token Usage** にて IP レンジを指定し、W&B インスタンスの IP や IP レンジを許可リストに加えることもできます。許可する IP アドレスを制限することで、Slack アプリケーションのセキュリティをさらに高めることができます。

## Slack アプリケーションを W&B に登録する

1. お使いの W&B インスタンスの **System Settings** または **System Console** ページに移動します（デプロイメントによります）

2. システムページに応じて、次のいずれかの操作を行います：

    - **System Console** の場合：**Settings** に進み **Notifications** を選択します。

      {{< img src="/images/hosting/register_slack_app_console.png" alt="System Console notifications" >}}

    - **System Settings** の場合：**Enable a custom Slack application to dispatch alerts** を切り替え、カスタム Slack アプリケーションを有効にします。

      {{< img src="/images/hosting/register_slack_app.png" alt="Slack アプリケーション有効化トグル" >}}

3. **Slack client ID** と **Slack secret** を入力し、**Save** をクリックします。アプリのクライアント ID やシークレットは Settings の Basic Information から確認できます。

4. W&B アプリで Slack インテグレーションを設定し、すべて正しく動作しているか確認してください。