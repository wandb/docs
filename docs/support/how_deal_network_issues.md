---
title: "How do I deal with network issues?"
tags:
   - None
---

If you're seeing SSL or network errors:`wandb: Network error (ConnectionError), entering retry loop`. You can try a couple of different approaches to solving this issue:

1. Upgrade your SSL certificate. If you're running the script on an Ubuntu server, run `update-ca-certificates` We can't sync training logs without a valid SSL certificate because it's a security vulnerability.
2. If your network is flaky, run training in [offline mode](../guides/track/launch.md) and sync the files to us from a machine that has Internet access.
3. Try running [W&B Private Hosting](../guides/hosting/intro.md), which operates on your machine and doesn't sync files to our cloud servers.

`SSL CERTIFICATE_VERIFY_FAILED`: this error could be due to your company's firewall. You can set up local CAs and then use:

`export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt`