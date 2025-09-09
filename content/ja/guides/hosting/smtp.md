---
title: SMTP を設定する
menu:
  default:
    identifier: ja-guides-hosting-smtp
    parent: w-b-platform
weight: 6
---

W&B サーバーでは、インスタンスまたは Teams に Users を追加すると、メール招待が送信されます。これらのメール招待の送信には、W&B はサードパーティのメールサーバーを使用します。場合によっては、組織で社内ネットワーク外へのトラフィックに厳格なポリシーを設けているため、メール招待がエンドユーザーに届かないことがあります。W&B サーバーは、内部 SMTP サーバー経由でこれらの招待メールを送信するように設定できるオプションを提供します。

設定するには、以下の手順に従ってください。

- `GORILLA_EMAIL_SINK` 環境変数を、Docker コンテナまたは Kubernetes デプロイメント内で `smtp://<user:password>@smtp.host.com:<port>` に設定します。
- `username` および `password` はオプションです。
- 認証不要な SMTP サーバーを使用している場合は、環境変数の値を `GORILLA_EMAIL_SINK=smtp://smtp.host.com:<port>` のように設定するだけで済みます。
- SMTP で一般的に使用されるポート番号は 587、465、25 です。使用しているメールサーバーの種類によって異なる場合がある点に注意してください。
- SMTP のデフォルトの送信元メールアドレス（初期値は `noreply@wandb.com`）は任意のメールアドレスに変更できます。サーバー上で `GORILLA_EMAIL_FROM_ADDRESS` 環境変数を希望する送信元メールアドレスに設定してください。