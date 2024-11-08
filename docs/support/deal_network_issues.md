---
title: "How do I deal with network issues?"
displayed_sidebar: support
tags:
   - connectivity
---
If you encounter SSL or network errors, such as `wandb: Network error (ConnectionError), entering retry loop`, use the following solutions:

1. Upgrade the SSL certificate. On an Ubuntu server, run `update-ca-certificates`. A valid SSL certificate is essential for syncing training logs to mitigate security risks.
2. If the network connection is unstable, operate in offline mode by setting the [optional environment variable](../guides/track/environment-variables/#optional-environment-variables) `WANDB_MODE` to `offline`, and sync files later from a device with Internet access.
3. Consider using [W&B Private Hosting](../guides/hosting/intro.md), which runs locally and avoids syncing to cloud servers.

For the `SSL CERTIFICATE_VERIFY_FAILED` error, this issue might stem from a company firewall. Configure local CAs and execute:

`export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt`