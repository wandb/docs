---
displayed_sidebar: default
---


# Slack alerts

W&B Server を [Slack](https://slack.com/) とインテグレーションします。

## Slackアプリケーションを作成する

以下の手順に従って、Slackアプリケーションを作成します。

1. https://api.slack.com/apps を訪問し、**Create an App** を選択します。

![](/images/hosting/create_an_app.png)

2. **App Name** フィールドにアプリの名前を入力します。
3. アプリを開発する Slack ワークスペースを選択します。アラートに使用する予定のワークスペースと同じワークスペースを選択してください。

![](/images/hosting/name_app_workspace.png)

## Slack アプリケーションの設定

1. 左サイドバーで **OAth & Permissions** を選択します。

![](/images/hosting/add_an_oath.png)

2. Scopes セクション内で、bot に **incoming_webhook** スコープを付与します。スコープは、開発ワークスペースでアプリケーションがアクションを実行する許可を与えます。

   Bots の OAuth スコープについての詳細は、Slack api ドキュメントの「Understanding OAuth scopes for Bots」チュートリアルを参照してください。

![](/images/hosting/save_urls.png)

3. Redirect URL を W&B インストール先に設定します。ローカルシステム設定でホストURLに設定されているのと同じURLを使用してください。インスタンスに異なるDNSマッピングがある場合は、複数のURLを指定できます。

![](/images/hosting/redirect_urls.png)

4. **Save URLs** を選択します。
5. 必要に応じて **Restrict API Token Usage** でIP範囲を指定し、W&BインスタンスのIPまたはIP範囲を許可リストに追加します。許可されるIPアドレスを制限することは、Slackアプリケーションのセキュリティをさらに強化するのに役立ちます。

## Slack アプリケーションを W&B に登録する

1. W&B インスタンスの **System Settings** ページに移動します。**Enable a custom Slack application to dispatch alerts** をトグルしてカスタム Slack アプリケーションの使用を有効にします。

![](/images/hosting/register_slack_app.png)

Slack アプリのクライアントIDとシークレットを入力する必要があります。設定の基本情報に移動して、アプリケーションのクライアントIDとシークレットを見つけてください。

2. W&B アプリで Slack インテグレーションを設定して、すべてが正常に動作することを確認します。