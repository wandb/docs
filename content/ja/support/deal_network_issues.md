---
title: How do I deal with network issues?
menu:
  support:
    identifier: ja-support-deal_network_issues
tags:
- connectivity
toc_hide: true
type: docs
---

SSL またはネットワークのエラー（`wandb: Network error (ConnectionError), entering retry loop`など）が発生した場合は、次の解決策をお試しください。

1. SSL 証明書をアップグレードします。Ubuntu サーバーで `update-ca-certificates` を実行します。有効な SSL 証明書は、セキュリティ リスクを軽減するために、トレーニング ログを同期するために不可欠です。
2. ネットワーク接続が不安定な場合は、[オプションの 環境 変数]({{< relref path="/guides/models/track/environment-variables.md#optional-environment-variables" lang="ja" >}}) `WANDB_MODE` を `offline` に設定してオフライン モードで動作させ、後でインターネット アクセスのあるデバイスからファイルを同期します。
3. ローカルで実行され、クラウド サーバーとの同期を回避する、[W&B Private Hosting]({{< relref path="/guides/hosting/" lang="ja" >}}) の使用を検討してください。

`SSL CERTIFICATE_VERIFY_FAILED` エラーの場合、この問題は会社のファイアウォールに起因する可能性があります。ローカル CA を構成して、以下を実行します。

`export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt`
