---
title: Catalyst
description: Catalyst, Pytorch 프레임워크에 W&B를 통합하는 방법
menu:
  default:
    identifier: ko-guides-integrations-catalyst
    parent: integrations
weight: 30
---

[Catalyst](https://github.com/catalyst-team/catalyst)는 PyTorch 기반의 딥러닝 R&D 프레임워크로, 재현성, 빠른 실험, 코드베이스 재사용에 중점을 두어 새로운 것을 만들 수 있도록 지원합니다.

Catalyst는 파라미터, 메트릭, 이미지 및 기타 아티팩트를 로그할 수 있는 W&B 인테그레이션을 제공합니다.

Python과 Hydra를 활용한 예제가 포함된 [인테그레이션 문서](https://catalyst-team.github.io/catalyst/api/loggers.html#catalyst.loggers.wandb.WandbLogger)를 참고해보세요.

## 인터랙티브 예제

[예제 colab](https://colab.research.google.com/drive/1PD0LnXiADCtt4mu7bzv7VfQkFXVrPxJq?usp=sharing)를 실행하여 Catalyst와 W&B 인테그레이션을 직접 확인해보세요.