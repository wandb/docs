---
title: Why is nothing showing up in my graphs?
menu:
  support:
    identifier: ko-support-kb-articles-graphs_nothing_showing
support:
- experiments
toc_hide: true
type: docs
url: /ko/support/:filename
translationKey: graphs_nothing_showing
---
"아직 시각화 데이터가 기록되지 않았습니다"라는 메시지가 표시되면 스크립트가 첫 번째 `wandb.log` 호출을 실행하지 않은 것입니다. 이러한 상황은 run이 단계를 완료하는 데 오랜 시간이 걸리는 경우에 발생할 수 있습니다. 데이터 로깅을 가속화하려면 에포크가 끝날 때 한 번만 로그하는 대신 에포크당 여러 번 로그하세요.
