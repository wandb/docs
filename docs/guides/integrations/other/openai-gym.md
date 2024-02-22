---
description: How to integrate W&B with OpenAI Gym.
slug: /guides/integrations/openai-gym
displayed_sidebar: default
---

# OpenAI Gym

:::info
"2021년부터 Gym을 유지 관리해온 팀은 모든 미래 개발을 [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)(import gymnasium as gym으로 Gym 대체)으로 옮겼으며, Gym은 더 이상의 업데이트를 받지 않을 것입니다." ([출처](https://github.com/openai/gym#the-team-that-has-been-maintaining-gym-since-2021-has-moved-all-future-development-to-gymnasium-a-drop-in-replacement-for-gym-import-gymnasium-as-gym-and-gym-will-not-be-receiving-any-future-updates-please-switch-over-to-gymnasium-as-soon-as-youre-able-to-do-so-if-youd-like-to-read-more-about-the-story-behind-this-switch-please-check-out-this-blog-post))

Gym이 더 이상 활발히 유지 관리되지 않으므로, Gymnasium과의 통합을 시도해 보세요. 여기에서 자세히 알아보세요 # TODO 링크 추가.
:::

[OpenAI Gym](https://gym.openai.com/)을 사용하는 경우 `gym.wrappers.Monitor`에 의해 생성된 환경의 비디오를 자동으로 기록합니다. [`wandb.init`](../../../ref/python/init.md)에 `monitor_gym` 키워드 인수를 `True`로 설정하거나 `wandb.gym.monitor()`를 호출하기만 하면 됩니다.

저희의 gym 통합은 매우 간단합니다. `gym`에서 기록된 비디오 파일의 이름을 [확인](https://github.com/wandb/wandb/blob/master/wandb/integration/gym/\_\_init\_\_.py#L15)하여 그 이름을 사용하거나 일치하는 것을 찾지 못할 경우 `"videos"`로 대체합니다. 더 많은 제어를 원한다면 언제든지 직접 [비디오 기록](../../track/log/media.md)을 할 수 있습니다.

[CleanRL](https://github.com/vwxyzjn/cleanrl)에 의한 [OpenRL Benchmark](http://wandb.me/openrl-benchmark-report)는 이 통합을 사용하여 OpenAI Gym 예제를 사용합니다. gym을 사용하는 방법을 보여주는 소스 코드(특정 실행에 사용된 [특정 코드 포함](https://wandb.ai/cleanrl/cleanrl.benchmark/runs/2jrqfugg/code?workspace=user-costa-huang))를 찾을 수 있습니다.

![여기에서 자세히 알아보세요: http://wandb.me/openrl-benchmark-report](/images/integrations/open_ai_report_example.png)