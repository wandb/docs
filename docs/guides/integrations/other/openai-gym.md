---
title: OpenAI Gym
description: OpenAI Gym과 W&B 통합 방법.
slug: /guides/integrations/openai-gym
displayed_sidebar: default
---

:::info
"2021년부터 Gym을 유지해 온 팀이 모든 향후 개발을 [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)으로 이전했습니다. 이는 Gym의 대체재로, Gym은 더 이상 업데이트를 받지 않을 것입니다." ([출처](https://github.com/openai/gym#the-team-that-has-been-maintaining-gym-since-2021-has-moved-all-future-development-to-gymnasium-a-drop-in-replacement-for-gym-import-gymnasium-as-gym-and-gym-will-not-be-receiving-any-future-updates-please-switch-over-to-gymnasium-as-soon-as-youre-able-to-do-so-if-youd-like-to-read-more-about-the-story-behind-this-switch-please-check-out-this-blog-post))

Gym은 더 이상 적극적으로 유지 관리되지 않는 프로젝트이므로, Gymnasium과의 인테그레이션을 시도해 보세요. 이에 대한 자세한 정보는 여기를 참고하세요. # TODO 링크 추가
:::

[OpenAI Gym](https://gym.openai.com/)을 사용하는 경우 `gym.wrappers.Monitor`에 의해 생성된 환경의 비디오를 자동으로 로그합니다. `monitor_gym` 키워드 인수를 [`wandb.init`](../../../ref/python/init.md)에 `True`로 설정하거나 `wandb.gym.monitor()`를 호출하면 됩니다.

우리의 gym 인테그레이션은 매우 가볍습니다. 우리는 단순히 `gym`에서 로그된 비디오 파일의 [이름을 확인](https://github.com/wandb/wandb/blob/master/wandb/integration/gym/__init__.py#L15)하여 그 이름을 따라가거나 일치하는 항목이 없을 경우 "videos"를 기본으로 사용합니다. 더 많은 제어를 원할 경우, 수동으로 [비디오를 로그](../../track/log/media.md)할 수 있습니다.

[CleanRL](https://github.com/vwxyzjn/cleanrl)의 [OpenRL Benchmark](http://wandb.me/openrl-benchmark-report)은 OpenAI Gym 예제를 위해 이 인테그레이션을 사용합니다. gym을 사용하는 방법을 보여주는 소스 코드 (특정 run에 사용된 [특정 코드](https://wandb.ai/cleanrl/cleanrl.benchmark/runs/2jrqfugg/code?workspace=user-costa-huang) 포함)를 찾을 수 있습니다.

![자세한 내용은 여기를 참조하세요: http://wandb.me/openrl-benchmark-report](/images/integrations/open_ai_report_example.png)