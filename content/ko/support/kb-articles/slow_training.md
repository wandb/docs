---
title: wandb가 내 트레이닝 속도를 느리게 만들까요?
menu:
  support:
    identifier: ko-support-kb-articles-slow_training
support:
- Experiments
toc_hide: true
type: docs
url: /support/:filename
---

W&B 는 일반적인 사용 조건에서 트레이닝 성능에 거의 영향을 주지 않습니다. 일반적인 사용에는 초당 한 번 미만의 로그 및 각 스텝마다 몇 메가바이트 이하의 데이터로 제한하는 것이 포함됩니다. W&B 는 별도의 프로세스에서 비동기 함수 호출로 동작하여, 일시적인 네트워크 장애나 간헐적인 디스크 읽기/쓰기 문제가 발생해도 성능이 저하되지 않도록 합니다. 지나치게 많은 데이터를 과도하게 로그할 경우 디스크 I/O 문제가 발생할 수 있습니다. 추가 문의 사항이 있으면 support 로 연락해 주세요.