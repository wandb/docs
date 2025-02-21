---
title: How does someone without an account see run results?
menu:
  support:
    identifier: ja-support-run_results_anonymous_mode
tags:
- anonymous
toc_hide: true
type: docs
---

`anonymous="allow"` でスクリプトを実行した場合：

1.  **一時アカウントの自動作成**: W&B はサインイン済みのアカウントを確認します。存在しない場合、W&B は新しい匿名アカウントを作成し、そのセッションの APIキー を保存します。
2.  **結果を迅速に ログ**: ユーザー はスクリプトを繰り返し実行し、W&B ダッシュボード で結果を即座に確認できます。これらの未請求の匿名 run は、7日間利用可能です。
3.  **データ が有用な場合に請求**: ユーザー が W&B で価値のある 結果 を特定したら、ページ上部のバナーにあるボタンをクリックして、run データ を実際のアカウントに保存できます。請求しない場合、run データ は7日後に削除されます。

{{% alert color="secondary" %}}
**匿名 run のリンクは機密情報です**。これらのリンクを使用すると、誰でも7日間 実験 結果を表示および請求できるため、信頼できる相手とのみリンクを共有してください。作成者の身元を隠しながら 結果 を公開で共有するには、support@wandb.com までお問い合わせください。
{{% /alert %}}

W&B ユーザー がスクリプトを見つけて実行すると、通常の run と同様に、 結果 はアカウントに正しく ログ されます。
