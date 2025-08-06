---
title: Farama Gymnasium
description: W&B 를 Farama Gymnasium과 연동하는 방법
menu:
  default:
    identifier: ko-guides-integrations-farama-gymnasium
    parent: integrations
weight: 90
---

[Farama Gymnasium](https://gymnasium.farama.org/#) 을 사용하고 있다면, `gymnasium.wrappers.Monitor` 로 생성된 환경의 비디오를 자동으로 로그합니다. [`wandb.init`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ko" >}}) 에서 `monitor_gym` 키워드 인수를 `True` 로 설정하기만 하면 됩니다.

Gymnasium 인테그레이션은 매우 간단합니다. 우리는 단순히 `gymnasium` 에서 로그되는 [비디오 파일의 이름을 확인](https://github.com/wandb/wandb/blob/c5fe3d56b155655980611d32ef09df35cd336872/wandb/integration/gym/__init__.py#LL69C67-L69C67) 해서, 그 이름을 사용하거나 일치하는 파일이 없으면 `"videos"` 로 기본 지정합니다. 더 세밀하게 제어하고 싶다면 [비디오를 직접 수동으로 로그]({{< relref path="/guides/models/track/log/media.md" lang="ko" >}}) 할 수도 있습니다.

Gymnasium 과 CleanRL 라이브러리를 함께 사용하는 방법은 이 [report](https://wandb.ai/raph-test/cleanrltest/reports/Mario-Bros-but-with-AI-Gymnasium-and-CleanRL---Vmlldzo0NTcxNTcw) 를 참고하세요.

{{< img src="/images/integrations/gymnasium.png" alt="Gymnasium RL environment" >}}