---
title: SMTP を設定
menu:
  default:
    identifier: ja-guides-hosting-smtp
    parent: w-b-platform
weight: 6
---

W&B サーバーでは、インスタンスやチームにユーザーを追加するとメール招待が送信されます。これらのメール招待を送信するために、W&B はサードパーティ製のメールサーバーを利用しています。組織によっては、社内ネットワーク外へのトラフィックに厳しいポリシーが設定されており、結果としてこれらのメール招待がユーザーに届かない場合があります。W&B サーバーでは、内部 SMTP サーバー経由で招待メールを送信するオプションも提供しています。

設定手順は以下の通りです。

- dockerコンテナや kubernetes デプロイメント内で `GORILLA_EMAIL_SINK` 環境変数を `smtp://<user:password>@smtp.host.com:<port>` に設定します
- `username` と `password` は任意です
- 認証不要の SMTP サーバーを使用する場合、環境変数の値を `GORILLA_EMAIL_SINK=smtp://smtp.host.com:<port>` のように設定してください
- SMTP でよく使われるポート番号は 587, 465, 25 です。ご利用されるメールサーバーの種類によって異なる場合がありますのでご注意ください。
- SMTP 用のデフォルト送信者メールアドレス（初期値は `noreply@wandb.com` です）を別のメールアドレスに変更したい場合は、サーバー上で `GORILLA_EMAIL_FROM_ADDRESS` 環境変数を希望の送信者メールアドレスに設定してください。