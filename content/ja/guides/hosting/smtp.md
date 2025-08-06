---
title: SMTP を設定
menu:
  default:
    identifier: smtp
    parent: w-b-platform
weight: 6
---

W&B サーバーでは、Users をインスタンスやTeam に追加すると、メール招待が送信されます。これらの招待メールを送信するために、W&B はサードパーティのメールサーバーを利用しています。場合によっては、組織のネットワーク外への通信に厳しいポリシーが設定されているため、招待メールがエンドユーザーに送信されないことがあります。W&B サーバーでは、内部の SMTP サーバーを使って招待メールを送信できるオプションを用意しています。

設定手順は以下の通りです。

- dockerコンテナ または kubernetes デプロイメントの環境変数 `GORILLA_EMAIL_SINK` に `smtp://<user:password>@smtp.host.com:<port>` を設定します
- `username` と `password` は省略可能です
- 認証不要の SMTP サーバーを利用する場合、環境変数の値を `GORILLA_EMAIL_SINK=smtp://smtp.host.com:<port>` のように設定してください
- SMTP でよく使われるポート番号は 587、465、25 です。ただし、利用するメールサーバーの種類によって異なる場合があります
- SMTP のデフォルト送信元メールアドレス（初期値は `noreply@wandb.com`）を任意のメールアドレスに変更したい場合は、サーバーの `GORILLA_EMAIL_FROM_ADDRESS` 環境変数に希望のアドレスを設定してください