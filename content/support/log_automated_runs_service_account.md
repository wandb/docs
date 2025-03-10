---
title: How do I log runs launched by continuous integration or internal tools?
toc_hide: true
type: docs
tags:
- runs
- logs
---
To launch automated tests or internal tools that log to W&B, create a **Service Account** on the team settings page. This action allows the use of a service API key for automated jobs, including those running through continuous integration. To attribute service account jobs to a specific user, set the `WANDB_USERNAME` or `WANDB_USER_EMAIL` environment variables.

{{< img src="/images/track/common_questions_automate_runs.png" alt="Create a service account on your team settings page for automated jobs" >}}