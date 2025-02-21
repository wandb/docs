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

SSL またはネットワークエラーが発生した場合、例えば `wandb: Network error (ConnectionError), entering retry loop` といったエラーが発生した場合は、以下の解決策を試してください。

1. SSL 証明書をアップグレードします。Ubuntu サーバーで `update-ca-certificates` を実行します。有効な SSL 証明書は、トレーニングログを同期してセキュリティリスクを軽減するために不可欠です。
2. ネットワーク接続が不安定な場合は、[任意の環境変数]({{< relref path="/guides/models/track/environment-variables.md#optional-environment-variables" lang="ja" >}}) `WANDB_MODE` を `offline` に設定し、オフラインモードで操作して、インターネット アクセスのあるデバイスから後でファイルを同期します。
3. ローカルで動作し、クラウド サーバーへの同期を回避する [W&B Private Hosting]({{< relref path="/guides/hosting/" lang="ja" >}}) の使用を検討してください。

`SSL CERTIFICATE_VERIFY_FAILED` エラーについて、この問題は会社のファイアウォールに起因する可能性があります。ローカルの CA を設定し、以下を実行します。

`export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt`