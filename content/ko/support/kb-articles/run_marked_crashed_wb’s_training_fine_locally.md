---
title: W&B에서 run 이 정상적으로 로컬에서 트레이닝 되고 있는데도 crashed로 표시되는 이유는 무엇인가요?
menu:
  support:
    identifier: ko-support-kb-articles-run_marked_crashed_wb’s_training_fine_locally
support:
- 크래시나 중단되는 run
toc_hide: true
type: docs
url: /support/:filename
---

이 메시지는 연결 문제를 나타냅니다. 서버가 인터넷 엑세스를 잃고 데이터가 W&B와 동기화되지 않으면, 시스템은 짧은 재시도 기간 후 해당 run 을 크래시된 것으로 표시합니다.