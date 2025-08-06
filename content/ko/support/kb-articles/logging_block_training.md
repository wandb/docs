---
title: 로그가 트레이닝을 방해하나요?
menu:
  support:
    identifier: ko-support-kb-articles-logging_block_training
support:
- 실험
toc_hide: true
type: docs
url: /support/:filename
---

"로그 함수는 지연(lazy) 동작인가요? 로컬 작업을 수행하는 동안 결과를 서버로 전송하기 위해 네트워크에 의존하고 싶지 않습니다."

`wandb.log` 함수는 로컬 파일에 한 줄을 기록하며, 네트워크 호출을 차단하지 않습니다. `wandb.init`을 호출하면 동일한 머신에서 새로운 프로세스가 시작됩니다. 이 프로세스는 파일 시스템 변화를 감지하고 웹 서비스와 비동기적으로 통신하여 로컬 작업이 중단 없이 계속될 수 있도록 합니다.