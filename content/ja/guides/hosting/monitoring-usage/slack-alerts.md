---
title: Configure Slack alerts
menu:
  default:
    identifier: ja-guides-hosting-monitoring-usage-slack-alerts
    parent: monitoring-and-usage
---

Integrate W&B Server with [Slack](https://slack.com/).

## Slack アプリケーションを作成する

次の手順に従って Slack のアプリケーションを作成します。

1. https://api.slack.com/apps にアクセスし、 **Create an App** を選択します。

    {{< img src="/images/hosting/create_an_app.png" alt="" >}}

2. **App Name** フィールドに、アプリの名前を入力します。
3. アプリを開発したい Slack ワークスペースを選択します。使用する Slack ワークスペースがアラート用に使用する予定のワークスペースと同じであることを確認してください。

    {{< img src="/images/hosting/name_app_workspace.png" alt="" >}}

## Slack アプリケーションを設定する

1. 左側のサイドバーで、 **OAth & Permissions** を選択します。

    {{< img src="/images/hosting/add_an_oath.png" alt="" >}}

2. Scopes セクション内で、ボットに **incoming_webhook** スコープを提供します。スコープは、開発ワークスペースで操作を実行するためのアプリの権限を与えます。

    Bot の OAuth スコープに関する詳細は、Slack API ドキュメントの "Understanding OAuth scopes for Bots" チュートリアルを参照してください。

    {{< img src="/images/hosting/save_urls.png" alt="" >}}

3. リダイレクト URL を W&B のインストール先を指すように設定します。ローカル システム設定のホスト URL 設定と同じ URL を使用します。インスタンスに異なる DNS マッピングがある場合は、複数の URL を指定できます。

    {{< img src="/images/hosting/redirect_urls.png" alt="" >}}

4. **Save URLs** を選択します。
5. 任意で **Restrict API Token Usage** の下に IP レンジを指定し、W&B インスタンスの IP または IP レンジを許可リスト化します。許可される IP アドレスを制限することで、Slack アプリケーションをさらに保護することができます。

## W&B に Slack アプリケーションを登録する

1. デプロイメントに応じて、W&B インスタンスの **System Settings** または **System Console** ページに移動します。

2. 現在のシステムページに応じて、以下のオプションのいずれかを実行します:

    - **System Console** にいる場合: **Settings** に移動し、次に **Notifications** に移動します。

      {{< img src="/images/hosting/register_slack_app_console.png" alt="" >}}

    - **System Settings** にいる場合: カスタム Slack アプリケーションでアラートを送信する **Enable a custom Slack application to dispatch alerts** をトグルして有効にします。

      {{< img src="/images/hosting/register_slack_app.png" alt="" >}}

3. **Slack client ID** と **Slack secret** を入力し、**Save** をクリックします。設定の Basic Information を参照して、アプリケーションのクライアント ID とシークレットを見つけます。

4. W&B アプリで Slack インテグレーションを設定して、すべてが正常に動作していることを確認します。