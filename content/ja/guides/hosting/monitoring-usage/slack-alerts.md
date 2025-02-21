---
title: Configure Slack alerts
menu:
  default:
    identifier: ja-guides-hosting-monitoring-usage-slack-alerts
    parent: monitoring-and-usage
---

W&B Server を [Slack](https://slack.com/) と連携させます。

## Slack アプリケーションの作成

以下の手順に従って Slack アプリケーションを作成してください。

1.  https://api.slack.com/apps にアクセスし、**Create an App** を選択します。

    {{< img src="/images/hosting/create_an_app.png" alt="" >}}

2.  **App Name** フィールドにアプリの名前を入力します。
3.  アプリの開発に使用する Slack ワークスペース を選択します。使用する Slack ワークスペース が、アラートに使用するワークスペース と同じであることを確認してください。

    {{< img src="/images/hosting/name_app_workspace.png" alt="" >}}

## Slack アプリケーションの設定

1.  左側のサイドバーで、**OAth & Permissions** を選択します。

    {{< img src="/images/hosting/add_an_oath.png" alt="" >}}

2.  Scopes セクションで、ボットに **incoming_webhook** スコープを付与します。スコープは、開発ワークスペース でアクションを実行するための権限をアプリに付与します。

    ボットの OAuth スコープの詳細については、Slack API ドキュメントの「ボットの OAuth スコープについて」チュートリアルを参照してください。

    {{< img src="/images/hosting/save_urls.png" alt="" >}}

3.  リダイレクト URL が W&B インストールを指すように設定します。ホスト URL がローカルシステム 設定で設定されている URL と同じ URL を使用します。インスタンスへの DNS マッピングが異なる場合は、複数の URL を指定できます。

    {{< img src="/images/hosting/redirect_urls.png" alt="" >}}

4.  **Save URLs** を選択します。
5.  オプションで、**Restrict API Token Usage** で IP 範囲を指定し、W&B インスタンスの IP または IP 範囲を許可リストに登録します。許可される IP アドレスを制限すると、Slack アプリケーションのセキュリティをさらに強化できます。

## W&B への Slack アプリケーションの登録

1.  デプロイメント に応じて、W&B インスタンスの **System Settings** または **System Console** ページに移動します。

2.  表示されているシステム ページに応じて、以下のいずれかのオプションに従います。

    -   **System Console** を使用している場合: **Settings** に移動し、次に **Notifications** に移動します。

      {{< img src="/images/hosting/register_slack_app_console.png" alt="" >}}

    -   **System Settings** を使用している場合: **Enable a custom Slack application to dispatch alerts** を切り替えて、カスタム Slack アプリケーションを有効にします。

      {{< img src="/images/hosting/register_slack_app.png" alt="" >}}

3.  **Slack client ID** と **Slack secret** を入力し、**Save** をクリックします。Settings の Basic Information に移動して、アプリケーションのクライアント ID とシークレットを確認します。

4.  W&B アプリで Slack インテグレーション を設定して、すべてが正常に動作することを確認します。
