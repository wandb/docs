---
description: How to integrate W&B with OpenAI Gym.
slug: /guides/integrations/openai-gym
displayed_sidebar: default
---

# OpenAI Gym

:::info
"2021년부터 Gym을 유지 관리해 온 팀은 모든 미래 개발을 Gym의 대체재인 [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)(import gymnasium as gym)으로 옮겼으며, Gym은 앞으로 어떤 업데이트도 받지 않을 예정입니다." ([출처](https://github.com/openai/gym#the-team-that-has-been-maintaining-gym-since-2021-has-moved-all-future-development-to-gymnasium-a-drop-in-replacement-for-gym-import-gymnasium-as-gym-and-gym-will-not-be-receiving-any-future-updates-please-switch-over-to-gymnasium-as-soon-as-youre-able-to-do-so-if-youd-like-to-read-more-about-the-story-behind-this-switch-please-check-out-this-blog-post))

Gym이 더 이상 활발하게 유지 관리되지 않기 때문에, Gymnasium과의 인테그레이션을 시도해 보세요. 여기에서 더 자세히 알아보세요 # TODO 링크 추가.
:::

[OpenAI Gym](https://gym.openai.com/)을 사용한다면, `gym.wrappers.Monitor`에 의해 생성된 환경의 비디오를 자동으로 로그합니다. [`wandb.init`](../../../ref/python/init.md)의 `monitor_gym` 키워드 인수를 `True`로 설정하거나 `wandb.gym.monitor()`를 호출하기만 하면 됩니다.

저희의 gym 인테그레이션은 매우 간단합니다. 우리는 `gym`에서 로그하는 비디오 파일의 이름을 [살펴보고](https://github.com/wandb/wandb/blob/master/wandb/integration/gym/__init__.py#L15) 그것을 이름으로 사용하거나 일치하는 것을 찾지 못하면 `"videos"`로 대체합니다. 더 많은 제어를 원한다면 언제든지 직접 [비디오 로그](../../track/log/media.md)를 할 수 있습니다.

[CleanRL](https://github.com/vwxyzjn/cleanrl)에 의한 [OpenRL 벤치마크](http://wandb.me/openrl-benchmark-report)는 이 인테그레이션을 OpenAI Gym 예제에 사용합니다. gym과 함께 사용하는 방법을 보여주는 소스 코드(특정 실행에 사용된 [특정 코드 포함](https://wandb.ai/cleanrl/cleanrl.benchmark/runs/2jrqfugg/code?workspace=user-costa-huang))를 찾을 수 있습니다.

![여기에서 더 알아보세요: http://wandb.me/openrl-benchmark-report](/images/integrations/open_ai_report_example.png)