---
description: How to integrate W&B with Farama Gymnasium.
menu:
  default:
    identifier: farama-gymnasium
    parent: integrations
title: Farama Gymnasium
weight: 90
---

If you're using [Farama Gymnasium](https://gymnasium.farama.org/#) we will automatically log videos of your environment generated by `gymnasium.wrappers.Monitor`. Just set the `monitor_gym` keyword argument to [`wandb.init`]({{< relref "/ref/python/sdk/functions/init.md" >}}) to `True`.

Our gymnasium integration is very light. We simply [look at the name of the video file](https://github.com/wandb/wandb/blob/c5fe3d56b155655980611d32ef09df35cd336872/wandb/integration/gym/__init__.py#LL69C67-L69C67) being logged from `gymnasium` and name it after that or fall back to `"videos"` if we don't find a match. If you want more control, you can always just manually [log a video]({{< relref "/guides/models/track/log/media.md" >}}).

Check out this [report](https://wandb.ai/raph-test/cleanrltest/reports/Mario-Bros-but-with-AI-Gymnasium-and-CleanRL---Vmlldzo0NTcxNTcw) to learn more on how to use Gymnasium with the CleanRL library. 

{{< img src="/images/integrations/gymnasium.png" alt="Gymnasium RL environment" >}}