---
title: 트레이닝 코드에서 run 이름을 어떻게 설정할 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-configure_name_run_training_code
support:
- Experiments
toc_hide: true
type: docs
url: /support/:filename
---

트레이닝 스크립트의 시작 부분에서, 실험 이름과 함께 `wandb.init`을 호출하세요. 예시: `wandb.init(name="my_awesome_run")`.