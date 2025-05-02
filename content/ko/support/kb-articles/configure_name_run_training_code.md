---
title: How can I configure the name of the run in my training code?
menu:
  support:
    identifier: ko-support-kb-articles-configure_name_run_training_code
support:
- experiments
toc_hide: true
type: docs
url: /ko/support/:filename
translationKey: configure_name_run_training_code
---
트레이닝 스크립트 시작 부분에서 실험 이름을 사용하여 `wandb.init`을 호출하세요. 예를 들어 `wandb.init(name="my_awesome_run")`과 같이 호출할 수 있습니다.
