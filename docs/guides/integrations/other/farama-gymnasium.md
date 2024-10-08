---
title: Farama Gymnasium
description: W&B와 Farama Gymnasium을 통합하는 방법.
slug: /guides/integrations/farama-gymnasium
displayed_sidebar: default
---

[Farama Gymnasium](https://gymnasium.farama.org/#)을 사용 중이라면 `gymnasium.wrappers.Monitor`에 의해 생성된 환경의 비디오를 자동으로 로그할 것입니다. `monitor_gym` 키워드 인수를 [`wandb.init`](../../../ref/python/init.md)에 `True`로 설정하세요.

우리의 gymnasium 인테그레이션은 매우 가볍습니다. 우리는 단순히 `gymnasium`에서 로그된 [비디오 파일의 이름을 확인](https://github.com/wandb/wandb/blob/c5fe3d56b155655980611d32ef09df35cd336872/wandb/integration/gym/__init__.py#LL69C67-L69C67)하고, 그 이름을 따르거나 일치하는 것을 찾지 못하면 기본값으로 `"videos"`를 사용합니다. 더 많은 제어가 필요하다면, 수동으로 [비디오를 로그](../../track/log/media.md)하는 것도 가능합니다.

CleanRL 라이브러리와 함께 Gymnasium을 사용하는 방법에 대해 더 알고 싶다면, 이 [report](https://wandb.ai/raph-test/cleanrltest/reports/Mario-Bros-but-with-AI-Gymnasium-and-CleanRL---Vmlldzo0NTcxNTcw)를 확인하세요.

![](/images/integrations/gymnasium.png)