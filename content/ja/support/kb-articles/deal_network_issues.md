---
title: ネットワークの問題にはどう対処すればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-deal_network_issues
support:
- 接続性
toc_hide: true
type: docs
url: /support/:filename
---

SSL や ネットワーク のエラー（例: `wandb: Network error (ConnectionError), entering retry loop`）が発生した場合は、次の対処法をお試しください:

1. SSL 証明書をアップグレードしてください。Ubuntu サーバーでは `update-ca-certificates` を実行します。有効な SSL 証明書は、セキュリティ リスクを軽減しつつトレーニング ログを同期するために不可欠です。
2. ネットワーク接続が不安定な場合は、[任意の環境変数]({{< relref path="/guides/models/track/environment-variables.md#optional-environment-variables" lang="ja" >}}) `WANDB_MODE` を `offline` に設定してオフラインモードで動作させ、インターネットに アクセス できる端末から後でファイルを同期してください。
3. ローカルで動作し、クラウド サーバーへの同期を回避できる [W&B Private Hosting]({{< relref path="/guides/hosting/" lang="ja" >}}) の利用を検討してください。

`SSL CERTIFICATE_VERIFY_FAILED` エラーの場合、企業のファイアウォールが原因の可能性があります。ローカル CA を設定し、次を実行してください:

`export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt`