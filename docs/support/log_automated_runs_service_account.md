---
title: How do I log runs launched by continuous integration or internal tools?
displayed_sidebar: support
tags:
- service account
- continuous integration
---
To launch automated tests or internal tools that log to W&B, create a **Service Account** on the team settings page. This action allows the use of a service API key for automated jobs, including those running through continuous integration. To attribute service account jobs to a specific user, set the `WANDB_USERNAME` or `WANDB_USER_EMAIL` environment variables.

![Create a service account on your team settings page for automated jobs](/images/track/common_questions_automate_runs.png)