---
title: Will wandb slow down my training?
menu:
  support:
    identifier: ko-support-slow_training
tags:
- experiments
toc_hide: true
type: docs
---

W&B는 일반적인 사용 조건에서 트레이닝 성능에 미치는 영향이 최소화됩니다. 일반적인 사용에는 초당 1회 미만의 빈도로 로깅하고 데이터를 스텝당 몇 메가바이트로 제한하는 것이 포함됩니다. W&B는 비차단 함수 호출을 통해 별도의 프로세스 에서 작동하므로, 짧은 네트워크 중단이나 간헐적인 디스크 읽기/쓰기 문제가 성능을 저해하지 않습니다. 대량의 데이터를 과도하게 로깅하면 디스크 I/O 문제가 발생할 수 있습니다. 더 궁금한 점이 있으시면 지원팀에 문의하십시오.
