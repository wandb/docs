---
description: How to integrate W&B with OpenAI Gym.
menu:
  default:
    identifier: openai-gym
    parent: integrations
title: OpenAI Gym
weight: 260
---

{{% alert %}}
"The team that has been maintaining Gym since 2021 has moved all future development to [Gymnasium](https://github.com/Farama-Foundation/Gymnasium), a drop in replacement for Gym (import gymnasium as gym), and Gym will not be receiving any future updates." ([Source](https://github.com/openai/gym#the-team-that-has-been-maintaining-gym-since-2021-has-moved-all-future-development-to-gymnasium-a-drop-in-replacement-for-gym-import-gymnasium-as-gym-and-gym-will-not-be-receiving-any-future-updates-please-switch-over-to-gymnasium-as-soon-as-youre-able-to-do-so-if-youd-like-to-read-more-about-the-story-behind-this-switch-please-check-out-this-blog-post))

Since Gym is no longer an actively maintained project, try out our integration with Gymnasium.
{{% /alert %}}

If you're using [OpenAI Gym](https://github.com/openai/gym), Weights & Biases automatically logs videos of your environment generated by `gym.wrappers.Monitor`. Just set the `monitor_gym` keyword argument to [`wandb.init`]({{< relref "/ref/python/sdk/functions/init.md" >}}) to `True` or call `wandb.gym.monitor()`.

Our gym integration is very light. We simply [look at the name of the video file](https://github.com/wandb/wandb/blob/master/wandb/integration/gym/__init__.py#L15) being logged from `gym` and name it after that or fall back to `"videos"` if we don't find a match. If you want more control, you can always just manually [log a video]({{< relref "/guides/models/track/log/media.md" >}}).

The [OpenRL Benchmark](https://wandb.me/openrl-benchmark-report) by[ CleanRL](https://github.com/vwxyzjn/cleanrl) uses this integration for its OpenAI Gym examples. You can find source code (including [the specific code used for specific runs](https://wandb.ai/cleanrl/cleanrl.benchmark/runs/2jrqfugg/code?workspace=user-costa-huang)) that demonstrates how to use gym with

{{< img src="/images/integrations/open_ai_report_example.png" alt="OpenAI Gym dashboard" >}}