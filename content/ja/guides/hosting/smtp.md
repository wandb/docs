---
title: Configure SMTP
menu:
  default:
    identifier: ja-guides-hosting-smtp
    parent: w-b-platform
weight: 6
---

W&B サーバーでは、インスタンスやチームにユーザーを追加すると、メール招待がトリガーされます。これらのメール招待を送信するために、W&B は外部のメールサーバーを使用します。場合によっては、組織が企業ネットワークから出るトラフィックに厳しいポリシーを持っており、そのためエンドユーザーにこれらのメール招待が送信されないことがあります。W&B サーバーは、内部の SMTP サーバー経由でこれらの招待メールを送信するオプションを提供しています。

設定するには、以下の手順に従ってください:

- `GORILLA_EMAIL_SINK` 環境変数を docker コンテナまたは kubernetes デプロイメント内で `smtp://<user:password>@smtp.host.com:<port>` に設定します。
- `username` と `password` はオプションです。
- 認証されていない SMTP サーバーを使用している場合、環境変数の値を `GORILLA_EMAIL_SINK=smtp://smtp.host.com:<port>` のように設定します。
- SMTP に一般的に使用されるポート番号は、ポート 587、465、および 25 です。ただし、使用するメールサーバーの種類に応じて異なる場合があります。
- SMTP のデフォルトの送信者メールアドレス（初期設定では `noreply@wandb.com`）を設定するには、お好みのメールアドレスに更新できます。これを行うには、サーバーで `GORILLA_EMAIL_FROM_ADDRESS` 環境変数を設定して希望する送信者メールアドレスにします。