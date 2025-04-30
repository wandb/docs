---
title: Does logging block my training?
menu:
  support:
    identifier: ko-support-kb-articles-logging_block_training
support:
- experiments
toc_hide: true
type: docs
url: /ko/support/:filename
---

"로깅 함수는 지연(lazy) 방식으로 작동하나요? 로컬 작업을 실행하는 동안 결과 를 서버 로 전송하기 위해 네트워크에 의존하고 싶지 않습니다."

`wandb.log` 함수는 로컬 파일에 한 줄을 쓰고 네트워크 호출을 차단하지 않습니다. `wandb.init`을 호출하면 동일한 머신에서 새 process 가 시작됩니다. 이 process 는 파일 시스템 변경을 수신하고 웹 서비스와 비동기적으로 통신하여 로컬 작업이 중단 없이 계속되도록 합니다.
