---
title: 만약 wandb가 크래시(오류) 나면, 내 트레이닝 run도 같이 중단될 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-crashes_crash_training_run
support:
- 크래시나 멈춤이 발생한 runs
toc_hide: true
type: docs
url: /support/:filename
---

트레이닝 run 에 방해를 주지 않는 것이 매우 중요합니다. W&B 는 별도의 프로세스에서 동작하므로, W&B 에 문제가 발생해도 트레이닝은 계속 진행됩니다. 인터넷 연결이 끊긴 경우에도, W&B 는 [wandb.ai](https://wandb.ai) 로 데이터를 보내기 위해 계속해서 재시도합니다.