---
title: Configure Slack alerts
menu:
  default:
    identifier: ja-guides-hosting-monitoring-usage-slack-alerts
    parent: monitoring-and-usage
---

W&B Server を [Slack](https://slack.com/) と連携させます。

{{% alert %}}
[W&B 専用クラウド デプロイメントで Slack アラートを設定するデモビデオ](https://www.youtube.com/watch?v=JmvKb-7u-oU) （6分）をご覧ください。
{{% /alert %}}

## Slack アプリケーションの作成

以下の手順に従って Slack アプリケーションを作成します。

1. https://api.slack.com/apps にアクセスし、**Create an App** を選択します。

    {{< img src="/images/hosting/create_an_app.png" alt="" >}}

2. **App Name** フィールドにアプリの名前を入力します。
3. アプリの開発に使用する Slack ワークスペースを選択します。使用する Slack ワークスペースが、アラートに使用するワークスペースと同じであることを確認してください。

    {{< img src="/images/hosting/name_app_workspace.png" alt="" >}}

## Slack アプリケーションの設定

1. 左側のサイドバーで、**OAth & Permissions** を選択します。

    {{< img src="/images/hosting/add_an_oath.png" alt="" >}}

2. Scopes セクションで、ボットに **incoming_webhook** スコープを付与します。スコープは、開発ワークスペースでアクションを実行するための権限をアプリに付与します。

    ボットの OAuth スコープの詳細については、Slack API ドキュメントのボットの OAuth スコープの理解に関するチュートリアルを参照してください。

    {{< img src="/images/hosting/save_urls.png" alt="" >}}

3. リダイレクト URL が W&B インストールを指すように設定します。ホスト URL がローカル システム 設定で設定されている URL と同じ URL を使用します。インスタンスへの DNS マッピングが異なる場合は、複数の URL を指定できます。

    {{< img src="/images/hosting/redirect_urls.png" alt="" >}}

4. **Save URLs** を選択します。
5. オプションで、**Restrict API Token Usage** で IP 範囲を指定し、W&B インスタンスの IP または IP 範囲を許可リストに登録できます。許可される IP アドレスを制限すると、Slack アプリケーションのセキュリティをさらに強化できます。

## W&B への Slack アプリケーションの登録

1. デプロイメントに応じて、W&B インスタンスの **System Settings** または **System Console** ページに移動します。

2. 表示されている System ページに応じて、以下のいずれかのオプションに従ってください。

    - **System Console** を使用している場合: **Settings** に移動し、次に **Notifications** に移動します。

      {{< img src="/images/hosting/register_slack_app_console.png" alt="" >}}

    - **System Settings** を使用している場合: **Enable a custom Slack application to dispatch alerts** を切り替えて、カスタム Slack アプリケーションを有効にします。

      {{< img src="/images/hosting/register_slack_app.png" alt="" >}}

3. **Slack client ID** と **Slack secret** を入力し、**Save** をクリックします。Settings の Basic Information に移動して、アプリケーションの client ID と secret を確認します。

4. W&B アプリで Slack インテグレーションを設定して、すべてが機能していることを確認します。
