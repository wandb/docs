---
description: The Prompts Quickstart shows how to visualise and debug the execution flow of your LLM chains and pipelines
displayed_sidebar: ja
---
# SMTP設定

W&Bサーバーでは、インスタンスやチームにユーザーを追加すると、メール招待が送信されます。これらのメール招待を送信するために、W&Bはサードパーティのメールサーバーを使用しています。一部の組織では、企業ネットワークからのトラフィックに厳しいポリシーがあるため、これらのメール招待がエンドユーザーに送信されないことがあります。W&Bサーバーでは、内部のSMTPサーバーを経由してこれらの招待メールを送信するように設定するオプションが用意されています。

設定するには、以下の手順に従ってください。

- DockerコンテナまたはKubernetesのデプロイメントで`GORILLA_EMAIL_SINK`環境変数を`smtp://<user:password>@smtp.host.com:<port>`に設定します。

- `username`と`password`はオプションです。

- 認証が不要なSMTPサーバーを使用している場合、環境変数の値を`GORILLA_EMAIL_SINK=smtp://smtp.host.com:<port>`のように設定します。

- SMTPでよく使われるポート番号は587、465、25です。ただし、使用しているメールサーバーの種類によって異なる場合がありますので、ご注意ください。