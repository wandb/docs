---
title: 코드를 테스트할 때 wandb 를 끌 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-testing_code_turn_off
support:
- 아티팩트
toc_hide: true
type: docs
url: /support/:filename
---

`wandb.init(mode="disabled")`을 사용하거나 `WANDB_MODE=disabled`로 설정하면 테스트 목적을 위해 W&B를 동작하지 않는(NOOP) 상태로 구성할 수 있습니다.

{{% alert %}}
`wandb.init(mode="disabled")`를 사용해도 W&B가 `WANDB_CACHE_DIR`에 Artifacts를 저장하는 것은 차단되지 않습니다.
{{% /alert %}}