---
title: How does someone without an account see run results?
menu:
  support:
    identifier: ja-support-kb-articles-run_results_anonymous_mode
support:
- anonymous
toc_hide: true
type: docs
url: /support/:filename
---

`anonymous="allow"` でスクリプトを実行すると:

1.  **一時アカウントの自動作成**: W&B はサインイン済みのアカウントを確認します。存在しない場合、W&B は新しい匿名アカウントを作成し、そのセッションの API キーを保存します。
2.  **結果を迅速に ログ**: ユーザーはスクリプトを繰り返し実行し、W&B ダッシュボード で結果を即座に表示できます。これらの未請求の匿名 Run は、7日間利用可能です。
3.  **データ が有用な場合に請求**: ユーザーが W&B で価値のある結果を特定したら、ページ上部のバナーにあるボタンをクリックして、Run データ を実際のアカウントに保存できます。請求がない場合、Run データ は7日後に削除されます。

{{% alert color="secondary" %}}
**匿名 Run のリンクは機密情報です**。これらのリンクを使用すると、誰でも7日間 実験 結果を表示および請求できるため、信頼できる人にのみリンクを共有してください。作成者のIDを隠しながら結果を公に共有するには、support@wandb.com までご連絡ください。
{{% /alert %}}

W&B ユーザーがスクリプトを見つけて実行すると、通常 Run と同様に、結果がアカウントに正しく ログ されます。
