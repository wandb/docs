---
title: wandb.Run.log()에 클래스 속성을 전달하면 어떻게 되나요?
menu:
  support:
    identifier: ko-support-kb-articles-pass_class_attribute_wandblog
support:
- 실험
toc_hide: true
type: docs
url: /support/:filename
---

클래스 속성을 `wandb.Run.log()` 에 전달하는 것은 피하세요. 속성은 네트워크 호출이 실행되기 전에 변경될 수 있습니다. 메트릭을 클래스 속성으로 저장할 때는 딥 카피를 사용하여, `wandb.Run.log()` 호출 시점의 속성과 로그된 메트릭 값이 일치하도록 하세요.