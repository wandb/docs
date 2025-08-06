---
title: ネットワークの問題にどう対応すればよいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- 接続
---

SSL やネットワークエラー（例: `wandb: Network error (ConnectionError), entering retry loop`）が発生した場合は、以下の解決策をお試しください。

1. SSL 証明書をアップグレードします。Ubuntu サーバーの場合は `update-ca-certificates` を実行してください。有効な SSL 証明書は、トレーニングログの同期時のセキュリティリスクを抑えるために不可欠です。
2. ネットワーク接続が不安定な場合は、[オプションの環境変数]({{< relref "/guides/models/track/environment-variables.md#optional-environment-variables" >}}) `WANDB_MODE` を `offline` に設定し、オフラインモードで作業した後、インターネットに接続できるデバイスからファイルを同期してください。
3. [W&B Private Hosting]({{< relref "/guides/hosting/" >}}) のご利用もご検討ください。ローカルで動作し、クラウドサーバーへの同期を回避できます。

`SSL CERTIFICATE_VERIFY_FAILED` エラーについては、社内ファイアウォールが原因の場合があります。ローカルの CA を設定し、次のコマンドを実行してください。

`export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt`