---
title: What happens if I pass a class attribute into wandb.log()?
menu:
  support:
    identifier: ko-support-pass_class_attribute_wandblog
tags:
- experiments
toc_hide: true
type: docs
---

`wandb.log()`에 클래스 속성을 전달하지 마세요. 속성은 네트워크 호출이 실행되기 전에 변경될 수 있습니다. 메트릭을 클래스 속성으로 저장할 때, 로깅된 메트릭이 `wandb.log()` 호출 시점의 속성 값과 일치하도록 깊은 복사(deep copy)를 사용하세요.
