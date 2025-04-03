---
title: Configure SMTP
menu:
  default:
    identifier: ja-guides-hosting-smtp
    parent: w-b-platform
weight: 6
---

W&B サーバー では、インスタンスまたは Team に ユーザー を追加すると、メールによる招待が送信されます。これらのメールによる招待を送信するために、W&B はサードパーティのメールサーバーを使用します。組織によっては、企業ネットワークから送信されるトラフィックに関する厳格なポリシーがあり、その結果、これらのメールによる招待がエンド ユーザー に送信されない場合があります。W&B サーバー には、社内の SMTP サーバー 経由でこれらの招待メールを送信するように構成するオプションがあります。

構成するには、以下の手順に従ってください。

- dockerコンテナ または kubernetes の デプロイメント で `GORILLA_EMAIL_SINK` 環境 変数 を `smtp://<user:password>@smtp.host.com:<port>` に設定します。
- `username` と `password` はオプションです。
- 認証を必要としないように設計された SMTP サーバー を使用している場合は、環境 変数 の 値 を `GORILLA_EMAIL_SINK=smtp://smtp.host.com:<port>` のように設定するだけです。
- SMTP で一般的に使用される ポート 番号 は、 ポート 587、465、および 25 です。これは、使用しているメール サーバー の種類によって異なる場合があることに注意してください。
- SMTP のデフォルトの送信者メール アドレス （最初は `noreply@wandb.com` に設定されています）を構成するには、サーバー 上の `GORILLA_EMAIL_FROM_ADDRESS` 環境 変数 を目的の送信者メール アドレス に更新します。