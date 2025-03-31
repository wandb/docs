---
menu:
  support:
    identifier: ja-support-kb-articles-deal_network_issues
support:
- connectivity
title: How do I deal with network issues?
toc_hide: true
type: docs
url: /support/:filename
---

If you encounter SSL or network errors, such as `wandb: Network error (ConnectionError), entering retry loop`, use the following solutions:

1. Upgrade the SSL certificate. On an Ubuntu server, run `update-ca-certificates`. A valid SSL certificate is essential for syncing training logs to mitigate security risks.
2. If the network connection is unstable, operate in offline mode by setting the [optional environment variable]({{< relref path="/guides/models/track/environment-variables.md#optional-environment-variables" lang="ja" >}}) `WANDB_MODE` to `offline`, and sync files later from a device with Internet access.
3. Consider using [W&B Private Hosting]({{< relref path="/guides/hosting/" lang="ja" >}}), which runs locally and avoids syncing to cloud servers.

For the `SSL CERTIFICATE_VERIFY_FAILED` error, this issue might stem from a company firewall. Configure local CAs and execute:

`export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt`