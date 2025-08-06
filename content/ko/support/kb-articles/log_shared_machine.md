---
title: 공유 머신에서 올바른 wandb 사용자로 로그하려면 어떻게 해야 하나요?
menu:
  support:
    identifier: ko-support-kb-articles-log_shared_machine
support:
- 로그
toc_hide: true
type: docs
url: /support/:filename
---

공유 머신을 사용할 때는 `WANDB_API_KEY` 환경 변수를 설정하여 run 이 올바른 WandB 계정에 로그되는지 확인하세요. 이 변수를 환경에 추가하면 로그인 시 올바른 자격 증명이 제공됩니다. 또는 스크립트에서 직접 환경 변수를 설정할 수도 있습니다.

`export WANDB_API_KEY=X` 코맨드를 실행하고, X 부분을 본인의 API 키로 교체하세요. 로그인한 사용자는 [wandb.ai/authorize](https://app.wandb.ai/authorize) 에서 자신의 API 키를 확인할 수 있습니다.