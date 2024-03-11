---
description: How to integrate W&B with Farama Gymnasium.
slug: /guides/integrations/farama-gymnasium
displayed_sidebar: default
---

# 파라마 체육관

[Farama Gymnasium](https://gymnasium.farama.org/#)을 사용하는 경우, `gymnasium.wrappers.Monitor`에 의해 생성된 환경의 비디오를 자동으로 로그합니다. 단지 [`wandb.init`](../../../ref/python/init.md)에 `monitor_gym` 키워드 인수를 `True`로 설정하면 됩니다.

우리의 체육관 인테그레이션은 매우 가볍습니다. 단순히 `gymnasium`에서 로그되는 비디오 파일의 이름을 [살펴보고](https://github.com/wandb/wandb/blob/c5fe3d56b155655980611d32ef09df35cd336872/wandb/integration/gym/__init__.py#LL69C67-L69C67) 그것에 따라 이름을 지정하거나 일치하는 것을 찾지 못할 경우 `"videos"`로 대체합니다. 더 많은 제어를 원한다면, 언제든지 직접 [비디오를 로그](../../track/log/media.md)할 수 있습니다.

CleanRL 라이브러리와 Gymnasium을 사용하는 방법에 대해 자세히 알아보려면 이 [리포트](https://wandb.ai/raph-test/cleanrltest/reports/Mario-Bros-but-with-AI-Gymnasium-and-CleanRL---Vmlldzo0NTcxNTcw)를 확인하세요.

![](/images/integrations/gymnasium.png)