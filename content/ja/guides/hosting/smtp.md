---
title: SMTP を設定
menu:
  default:
    identifier: ja-guides-hosting-smtp
    parent: w-b-platform
weight: 6
---

W&B server では、 インスタンスやチームにユーザーを追加すると、メール招待がトリガーされます。これらのメール招待を送信するために、 W&B はサードパーティのメールサーバーを使用します。場合によっては、企業ネットワークからのトラフィックを厳しく制限するポリシーがあり、その結果としてこれらのメール招待がエンドユーザーに送信されないことがあります。W&B server は、内部 SMTP サーバーを通じてこれらの招待メールを送信するオプションを提供しています。

設定手順は次の通りです：

- dockerコンテナまたは kubernetes デプロイメント内で `GORILLA_EMAIL_SINK` 環境変数を `smtp://<user:password>@smtp.host.com:<port>` に設定します
- `username` と `password` はオプションです
- 認証不要な SMTP サーバーを使用している場合は、環境変数の値を `GORILLA_EMAIL_SINK=smtp://smtp.host.com:<port>` として設定します
- SMTPで一般的に使用されるポート番号は 587, 465, 25 です。お使いのメールサーバーの種類に応じて異なる場合がありますので注意してください。
- SMTP のデフォルト送信元メールアドレスを設定するには、 `GORILLA_EMAIL_FROM_ADDRESS` 環境変数を サーバー上であなたの望む送信元メールアドレスに設定することができます。初期設定は `noreply@wandb.com` になっていますが、これを変更することが可能です。