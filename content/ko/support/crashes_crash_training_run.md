---
title: If wandb crashes, will it possibly crash my training run?
menu:
  support:
    identifier: ko-support-crashes_crash_training_run
tags:
- crashing and hanging runs
toc_hide: true
type: docs
---

트레이닝 run과의 간섭을 피하는 것이 중요합니다. W&B는 별도의 process에서 작동하므로 W&B에 오류가 발생하더라도 트레이닝이 계속됩니다. 인터넷 연결이 끊긴 경우 W&B는 [wandb.ai](https://wandb.ai)로 데이터를 계속해서 다시 보냅니다.
