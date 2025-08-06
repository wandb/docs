---
title: ネットワークの問題に対処するにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-deal_network_issues
support:
- 接続
toc_hide: true
type: docs
url: /support/:filename
---

SSL やネットワークエラー（例：`wandb: Network error (ConnectionError), entering retry loop`）が発生した場合、以下の解決策をお試しください。

1. SSL 証明書をアップグレードします。Ubuntu サーバーでは `update-ca-certificates` を実行してください。正しい SSL 証明書は、 トレーニングログを安全に同期するために必要です。
2. ネットワーク接続が不安定な場合は、 [オプションの環境変数]({{< relref path="/guides/models/track/environment-variables.md#optional-environment-variables" lang="ja" >}}) `WANDB_MODE` を `offline` に設定してオフラインモードで作業し、後でインターネット接続のあるデバイスからファイルを同期してください。
3. ローカルで動作し、クラウドサーバーとの同期が不要な [W&B Private Hosting]({{< relref path="/guides/hosting/" lang="ja" >}}) の利用を検討してください。

`SSL CERTIFICATE_VERIFY_FAILED` エラーの場合、会社のファイアウォールが原因の可能性があります。ローカル CA を設定し、以下を実行してください。

`export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt`