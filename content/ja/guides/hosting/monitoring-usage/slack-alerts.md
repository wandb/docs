---
title: Slack アラートを設定
menu:
  default:
    identifier: ja-guides-hosting-monitoring-usage-slack-alerts
    parent: monitoring-and-usage
---

W&B サーバーを [Slack](https://slack.com/) と連携する。
{{% alert %}}
[W&B 専用クラウド デプロイメントでの Slack アラートの設定手順を紹介する動画](https://www.youtube.com/watch?v=JmvKb-7u-oU) をご覧ください（6 分）。
{{% /alert %}}

## Slack アプリケーションを作成する

以下の手順に従って Slack アプリケーションを作成します。

1. https://api.slack.com/apps にアクセスし、**Create an App** を選択します。

    {{< img src="/images/hosting/create_an_app.png" alt="Create an App ボタン" >}}

2. **App Name** フィールドにアプリの名前を入力します。
3. アプリを開発する Slack ワークスペースを選択します。アラートに使用するワークスペースと同じ Slack ワークスペースを選んでください。

    {{< img src="/images/hosting/name_app_workspace.png" alt="アプリ名とワークスペースの選択" >}}

## Slack アプリケーションを設定する

1. 左側のサイドバーで **OAuth & Permissions** を選択します。

    {{< img src="/images/hosting/add_an_oath.png" alt="OAuth & Permissions メニュー" >}}

2. Scopes セクションで、ボットに **incoming_webhook** スコープを付与します。スコープは、開発用ワークスペースでアプリが操作を行うための権限です。

    Bots の OAuth スコープの詳細は、Slack API ドキュメントの「Understanding OAuth scopes for Bots」チュートリアルを参照してください。

    {{< img src="/images/hosting/save_urls.png" alt="Bot トークンのスコープ" >}}

3. Redirect URL を W&B のインストール先を指すように設定します。ローカルのシステム設定でホスト URL に設定したものと同じ URL を使用してください。インスタンスに対して異なる DNS マッピングがある場合は、複数の URL を指定できます。

    {{< img src="/images/hosting/redirect_urls.png" alt="Redirect URLs の設定" >}}

4. **Save URLs** を選択します。
5. 必要に応じて、**Restrict API Token Usage** で IP 範囲を指定し、W&B インスタンスの IP または IP 範囲を許可リストに追加します。許可する IP アドレスを制限することで、Slack アプリケーションのセキュリティをさらに高められます。

## W&B に Slack アプリケーションを登録する

1. デプロイメントに応じて、W&B インスタンスの **System Settings** または **System Console** ページに移動します。

2. 表示しているシステムのページに応じて、以下のいずれかの手順に従います:

    - **System Console** の場合: **Settings** に移動し、続いて **Notifications** を開きます

      {{< img src="/images/hosting/register_slack_app_console.png" alt="System Console の通知" >}}

    - **System Settings** の場合: **Enable a custom Slack application to dispatch alerts** を切り替えて、カスタム Slack アプリケーションを有効にします

      {{< img src="/images/hosting/register_slack_app.png" alt="Slack アプリケーションの有効化トグル" >}}

3. **Slack client ID** と **Slack secret** を入力し、**Save** をクリックします。クライアント ID とシークレットは、Settings の Basic Information で確認できます。

4. W&B アプリで Slack インテグレーションを設定し、正しく動作することを確認します。