---
title: Slack アラートを設定
menu:
  default:
    identifier: ja-guides-hosting-monitoring-usage-slack-alerts
    parent: monitoring-and-usage
---

W&B サーバーを [Slack](https://slack.com/) と連携します。
{{% alert %}}
[W&B 専用クラウド デプロイメントで Slack アラートを設定するデモ動画](https://www.youtube.com/watch?v=JmvKb-7u-oU)（6分）をご覧ください。
{{% /alert %}}

## Slack アプリケーションの作成

以下の手順に従って、Slack アプリケーションを作成します。

1. https://api.slack.com/apps にアクセスし、**Create an App** を選択します。

    {{< img src="/images/hosting/create_an_app.png" alt="Create an App ボタン" >}}

2. **App Name** フィールドにアプリの名前を入力します。
3. アプリを開発したい Slack ワークスペースを選択します。アラートで利用したいワークスペースと同じものを選択してください。

    {{< img src="/images/hosting/name_app_workspace.png" alt="アプリ名とワークスペースの選択" >}}

## Slack アプリケーションの設定

1. 左サイドバーから **OAth & Permissions** を選択します。

    {{< img src="/images/hosting/add_an_oath.png" alt="OAuth & Permissions メニュー" >}}

2. Scopes セクションで **incoming_webhook** スコープを Bot に付与します。スコープによって、開発用ワークスペースでアプリが実行できる操作が決まります。

    Bot の OAuth スコープについての詳細は、Slack API ドキュメントの "Understanding OAuth scopes for Bots" チュートリアルをご確認ください。

    {{< img src="/images/hosting/save_urls.png" alt="Bot token スコープ" >}}

3. Redirect URL をあなたの W&B インストール先に設定します。ローカルシステムの設定で host URL として使っているものと同じ URL を指定してください。インスタンスへの DNS マッピングが複数ある場合は、それぞれの URL を指定できます。

    {{< img src="/images/hosting/redirect_urls.png" alt="Redirect URLs の設定" >}}

4. **Save URLs** を選択します。
5. オプションで、**Restrict API Token Usage** で IP 範囲を指定し、W&B インスタンスの IP または IP 範囲を許可リストに追加できます。許可された IP アドレスのみに絞ることで、Slack アプリケーションのセキュリティがさらに高まります。

## Slack アプリケーションを W&B に登録する

1. あなたの W&B インスタンスの **System Settings** または **System Console** ページへ移動します（ご利用環境によって異なります）。

2. 開いているシステムページによって、以下のどちらかの手順に従ってください。

    - **System Console** にいる場合：**Settings** から **Notifications** に進みます。

      {{< img src="/images/hosting/register_slack_app_console.png" alt="System Console の通知設定" >}}

    - **System Settings** にいる場合：**Enable a custom Slack application to dispatch alerts** を有効に切り替えてカスタム Slack アプリケーションを有効化します。

      {{< img src="/images/hosting/register_slack_app.png" alt="Slack アプリケーション有効化のトグル" >}}

3. **Slack client ID** と **Slack secret** を入力し、**Save** をクリックします。Settings の Basic Information でアプリケーションの client ID と secret を確認できます。

4. W&B アプリで Slack インテグレーションを設定し、すべてが正しく動作するか確認してください。