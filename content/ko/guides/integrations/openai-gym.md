---
title: OpenAI Gym
description: W&B를 OpenAI Gym과 통합하는 방법.
menu:
  default:
    identifier: ko-guides-integrations-openai-gym
    parent: integrations
weight: 260
---

{{% alert %}}
"2021년부터 Gym을 유지 관리해 온 팀은 향후 모든 개발을 Gym의 대체품인 [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)으로 이전했으며(gymnasium을 gym으로 가져오기), Gym은 더 이상 업데이트를 받지 않습니다." ([출처](https://github.com/openai/gym#the-team-that-has-been-maintaining-gym-since-2021-has-moved-all-future-development-to-gymnasium-a-drop-in-replacement-for-gym-import-gymnasium-as-gym-and-gym-will-not-be-receiving-any-future-updates-please-switch-over-to-gymnasium-as-soon-as-youre-able-to-do-so-if-youd-like-to-read-more-about-the-story-behind-this-switch-please-check-out-this-blog-post))

Gym은 더 이상 활발하게 유지 관리되는 프로젝트가 아니므로 Gymnasium과의 통합을 사용해 보십시오.
{{% /alert %}}

[OpenAI Gym](https://github.com/openai/gym)을 사용하는 경우, Weights & Biases는 `gym.wrappers.Monitor`에 의해 생성된 환경 비디오를 자동으로 기록합니다. `monitor_gym` 키워드 인수를 [`wandb.init`]({{< relref path="/ref/python/init.md" lang="ko" >}})에 `True`로 설정하거나 `wandb.gym.monitor()`를 호출하기만 하면 됩니다.

저희 gym integration은 매우 가볍습니다. `gym`에서 기록된 [비디오 파일의 이름을 확인](https://github.com/wandb/wandb/blob/master/wandb/integration/gym/__init__.py#L15)하여 그에 따라 이름을 지정하거나 일치하는 항목을 찾지 못하면 `"videos"`로 대체합니다. 더 많은 제어를 원하시면 언제든지 수동으로 [비디오를 기록]({{< relref path="/guides/models/track/log/media.md" lang="ko" >}})할 수 있습니다.

[CleanRL](https://github.com/vwxyzjn/cleanrl)의 [OpenRL Benchmark](http://wandb.me/openrl-benchmark-report)는 OpenAI Gym 예제에 이 integration을 사용합니다. gym과 함께 사용하는 방법을 보여주는 소스 코드([특정 run에 사용된 특정 코드](https://wandb.ai/cleanrl/cleanrl.benchmark/runs/2jrqfugg/code?workspace=user-costa-huang) 포함)를 찾을 수 있습니다.

{{< img src="/images/integrations/open_ai_report_example.png" alt="Learn more here: http://wandb.me/openrl-benchmark-report" >}}
