---
title: 왜 그래프에 아무것도 표시되지 않나요?
menu:
  support:
    identifier: ko-support-kb-articles-graphs_nothing_showing
support:
- 실험
toc_hide: true
type: docs
url: /support/:filename
---

"No visualization data logged yet"라는 메시지가 표시된다면, 스크립트가 첫 번째 `wandb.log` 호출을 실행하지 않은 것입니다. run 이 한 스텝을 완료하는 데 오랜 시간이 걸릴 때 이런 상황이 발생할 수 있습니다. 데이터 로그를 더 빠르게 남기려면 에포크가 끝난 후 한 번만 로그를 남기는 대신, 에포크당 여러 번 로그를 남겨보세요.