---
title: OpenAI Gym
description: W&B를 OpenAI Gym과 연동하는 방법
menu:
  default:
    identifier: ko-guides-integrations-openai-gym
    parent: integrations
weight: 260
---

{{% alert %}}
"2021년부터 Gym을 관리해 온 팀은 앞으로의 모든 개발을 [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)으로 이전했습니다. Gymnasium은 Gym의 바로 사용할 수 있는 대체재입니다(import gymnasium as gym). Gym은 더 이상 업데이트되지 않습니다." ([출처](https://github.com/openai/gym#the-team-that-has-been-maintaining-gym-since-2021-has-moved-all-future-development-to-gymnasium-a-drop-in-replacement-for-gym-import-gymnasium-as-gym-and-gym-will-not-be-receiving-any-future-updates-please-switch-over-to-gymnasium-as-soon-as-youre-able-to-do-so-if-youd-like-to-read-more-about-the-story-behind-this-switch-please-check-out-this-blog-post))

이제 Gym이 더 이상 적극적으로 유지관리되는 프로젝트가 아니기 때문에, 저희의 Gymnasium 인테그레이션을 사용해 보시기 바랍니다.
{{% /alert %}}

[OpenAI Gym](https://github.com/openai/gym)을 사용 중이라면, W&B는 `gym.wrappers.Monitor`로 생성된 환경의 비디오를 자동으로 로그합니다. [`wandb.init`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ko" >}})에서 `monitor_gym` 키워드 인수를 `True`로 설정하거나, 혹은 `wandb.gym.monitor()`를 호출하시면 됩니다.

Gym 인테그레이션은 매우 간단하게 동작합니다. W&B는 [로그되는 비디오 파일 이름을 확인](https://github.com/wandb/wandb/blob/master/wandb/integration/gym/__init__.py#L15)하여 해당 이름을 사용하거나, 매칭되는 이름이 없을 경우 `"videos"`로 대체해서 저장합니다. 만약 더 세부적으로 제어하고 싶으시다면 언제든 직접 [비디오를 수동으로 로그]({{< relref path="/guides/models/track/log/media.md" lang="ko" >}})하실 수 있습니다.

[OpenRL Benchmark](https://wandb.me/openrl-benchmark-report)는 [CleanRL](https://github.com/vwxyzjn/cleanrl)에서 제공되며, OpenAI Gym 예제를 실행할 때 이 인테그레이션을 사용합니다. 예제 코드를 포함하여 ([특정 run에 사용된 코드](https://wandb.ai/cleanrl/cleanrl.benchmark/runs/2jrqfugg/code?workspace=user-costa-huang)도 포함) Gym 사용 예시를 확인하실 수 있습니다.

{{< img src="/images/integrations/open_ai_report_example.png" alt="OpenAI Gym 대시보드" >}}