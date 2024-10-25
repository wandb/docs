
---
title: How can I resolve the Filestream rate limit exceeded error?
displayed_sidebar: support
tags:
- frontend
- backend
---
To resolve the "Filestream rate limit exceeded" error in Weights & Biases (W&B), try the following:

- **Optimize Logging**:
  - Reduce the frequency of logging or batch logs to minimize API requests.
  - Consider staggering experiment start times to avoid simultaneous API requests.

- **Check for outages**:
  - Ensure the issue is not due to a temporary server-side problem by checking [W&B status updates](https://status.wandb.com).

- **Contact Support**:
  - Reach out to W&B support with your experimental setup details to request an increase in rate limits.
    