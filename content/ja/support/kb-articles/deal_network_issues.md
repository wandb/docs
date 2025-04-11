---
title: ネットワークの問題にどのように対処すればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-deal_network_issues
support:
- connectivity
toc_hide: true
type: docs
url: /support/:filename
---

SSL またはネットワークエラーが発生した場合、例えば `wandb: Network error (ConnectionError), entering retry loop` というエラーが表示される場合、次の解決策を試してください。

1. SSL 証明書をアップグレードします。Ubuntu サーバーでは、`update-ca-certificates` を実行します。有効な SSL 証明書は、トレーニングログを同期してセキュリティリスクを軽減するために不可欠です。
2. ネットワーク接続が不安定な場合は、[オプションの環境変数]({{< relref path="/guides/models/track/environment-variables.md#optional-environment-variables" lang="ja" >}}) `WANDB_MODE` を `offline` に設定してオフラインモードで操作し、後でインターネットにアクセス可能なデバイスからファイルを同期します。
3. [W&B Private Hosting]({{< relref path="/guides/hosting/" lang="ja" >}}) の利用を検討してください。これによりローカルで実行し、クラウドサーバーへの同期を回避できます。

`SSL CERTIFICATE_VERIFY_FAILED` エラーについて、この問題は企業のファイアウォールに起因する可能性があります。ローカルの CA を設定して、次を実行します。

`export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt`