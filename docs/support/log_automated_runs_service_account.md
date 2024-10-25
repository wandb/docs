---
title: How do I log runs launched by continuous integration or internal tools?
displayed_sidebar: support
tags:
- service account
- continuous integration
---
If you have automated tests or internal tools that launch runs logging to W&B, create a **Service Account** on your team settings page. This will allow you to use a service API key for your automated jobs, such as those running via continuous integration. If you want to attribute service account jobs to a specific user, you can use the `WANDB_USERNAME` or `WANDB_USER_EMAIL` environment variables.

![Create a service account on your team settings page for automated jobs](/images/track/common_questions_automate_runs.png)