---
title: Can I turn off wandb when testing my code?
menu:
  support:
    identifier: ko-support-kb-articles-testing_code_turn_off
support:
- artifacts
toc_hide: true
type: docs
url: /support/:filename
---

테스트 목적으로 W&B를 작동하지 않는(NOOP) 상태로 구성하려면 `wandb.init(mode="disabled")`를 사용하거나 `WANDB_MODE=disabled`를 설정하세요.

{{% alert %}}
`wandb.init(mode="disabled")`를 사용해도 W&B가 아티팩트를 `WANDB_CACHE_DIR`에 저장하는 것을 막을 수는 없습니다.
{{% /alert %}}
