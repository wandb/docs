---
description: The Prompts Quickstart shows how to visualise and debug the execution flow of your LLM chains and pipelines
---
# Slackアラート

W&Bサーバーを[Slack](https://slack.com/)と連携させます。

## Slackアプリケーションの作成

以下の手順に従って、Slackアプリケーションを作成してください。

1. https://api.slack.com/appsにアクセスし、**アプリを作成**を選択します。

![](/images/hosting/create_an_app.png)

2. **アプリ名**欄にアプリの名前を入力してください。
3. アプリを開発するSlackワークスペースを選択します。アラートに使用する予定のワークスペースと同じものを使用してください。

![](/images/hosting/name_app_workspace.png)

## Slackアプリケーションの設定

1. 左のサイドバーで**OAuth & Permissions**を選択します。

![](/images/hosting/add_an_oath.png)

2. Scopesのセクション内で、ボットに**incoming_webhook**スコープを付与します。スコープは、開発ワークスペースでアプリがアクションを実行するための権限を与えます。

  ボット用のOAuthスコープについての詳細は、Slack APIドキュメントのUnderstanding OAuth scopes for Botsチュートリアルを参照してください。
  
![](/images/hosting/save_urls.png)

3. リダイレクトURLをW&Bインストールに設定します。ローカルシステム設定のホストURLに設定されているのと同じURLを使用してください。インスタンスに異なるDNSマッピングがある場合は、複数のURLを指定することができます。
![](/images/hosting/redirect_urls.png)

4. **URLを保存**を選択します。

5. 必要に応じて、**APIトークンの使用制限**の下でIP範囲を指定し、W&BインスタンスのIPまたはIP範囲を許可リストに追加できます。許可されたIPアドレスを制限することで、Slackアプリケーションのセキュリティをさらに強化できます。

## SlackアプリケーションをW&Bに登録する

1. W&Bインスタンスの**システム設定**ページに移動します。**カスタムSlackアプリケーションでアラートを送信する**を有効にして、カスタムSlackアプリケーションを有効にします：

![](/images/hosting/register_slack_app.png)

SlackアプリケーションのクライアントIDとシークレットを入力する必要があります。SettingsのBasic Informationに移動して、アプリケーションのクライアントIDとシークレットを見つけてください。

2. W&BアプリでSlackインテグレーションを設定して、すべてが正常に動作していることを確認します。