---
title: Limits
sidebar_label: Limits
---

## Rate Limits

The W&B API is rate limited by IP and API key.  New accounts are restricted to 200 requests per minute.  This rate allows you to run approximately 15 processes in parallel and have them report without being throttled.  If the **wandb** client detects it's being limited, it will backoff and retry sending the data in the future.  If you need to run more than 15 processes in parallel send an email to <a href="mailto:contact@wandb.com">contact@wandb.com</a>.

## Size Limits

### Files

The maximum file size for new accounts is 2GB.  A single run is allowed to store 10 GB of data.  If you need to store larger files or more data per run, contact us at <a href="mailto:contact@wandb.com">contact@wandb.com</a>.

### Metrics

Metrics are sampled to 500 data points by default before displaying in the UI.  Generally you shouldn't be calling `wandb.log` more than a few times per second or wandb may start to interfere with your training run's performance.

### Logs

While a run is in progress we tail the last 5000 lines of your log for you in the UI.  After a run is completed the entire log is archived and can be downloaded from an individual run page.