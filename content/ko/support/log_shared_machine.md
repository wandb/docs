---
title: How do I log to the right wandb user on a shared machine?
menu:
  support:
    identifier: ko-support-log_shared_machine
tags:
- logs
toc_hide: true
type: docs
---

공유 머신을 사용할 때는 인증을 위해 `WANDB_API_KEY` 환경 변수를 설정하여 run이 올바른 WandB 계정에 로그되도록 하세요. 환경에서 제공되는 경우, 이 변수는 로그인 시 올바른 자격 증명을 제공합니다. 또는 스크립트에서 직접 환경 변수를 설정하세요.

`export WANDB_API_KEY=X` 코맨드를 실행하고, X를 API 키로 대체하세요. 로그인한 사용자는 [wandb.ai/authorize](https://app.wandb.ai/authorize)에서 API 키를 찾을 수 있습니다.
