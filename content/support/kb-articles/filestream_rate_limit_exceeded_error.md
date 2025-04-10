---
url: /support/:filename
title: How can I resolve the Filestream rate limit exceeded error?
toc_hide: true
type: docs
support:
- connectivity
- outage
---
To resolve the "Filestream rate limit exceeded" error in Weights & Biases (W&B), follow these steps:

**Optimize logging**:
  - Reduce logging frequency or batch logs to decrease API requests.
  - Stagger experiment start times to avoid simultaneous API requests.

**Check for outages**:
  - Verify that the issue does not arise from a temporary server-side problem by checking [W&B status updates](https://status.wandb.com).

**Contact support**:
  - Reach out to W&B support (support@wandb.com) with details of the experimental setup to request an increase in rate limits.