---
title: How do I log to the right wandb user on a shared machine?
menu:
  support:
    identifier: ko-support-kb-articles-log_shared_machine
support:
- logs
toc_hide: true
type: docs
url: /support/:filename
---

공유 머신을 사용하는 경우, 인증을 위해 `WANDB_API_KEY` 환경 변수를 설정하여 run이 올바른 WandB 계정으로 로그되도록 하십시오. 환경에서 소싱된 경우, 이 변수는 로그인 시 올바른 자격 증명을 제공합니다. 또는 스크립트에서 직접 환경 변수를 설정하십시오.

`export WANDB_API_KEY=X` 코맨드를 실행하고, X를 사용자의 API 키로 교체합니다. 로그인한 사용자는 [wandb.ai/authorize](https://app.wandb.ai/authorize)에서 API 키를 찾을 수 있습니다.
