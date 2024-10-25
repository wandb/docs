---
title: How do I log to the right wandb user on a shared machine?
tags:
- logs
---
If you're using a shared machine and another person is a wandb user, it's easy to make sure your runs are always logged to the proper account. Set the `WANDB_API_KEY` environment variable to authenticate. If you source it in your env, when you log in you'll have the right credentials, or you can set the environment variable from your script.

Run this command `export WANDB_API_KEY=X` where X is your API key. When you're logged in, you can find your API key at [wandb.ai/authorize](https://app.wandb.ai/authorize).
