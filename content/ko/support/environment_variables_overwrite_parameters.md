---
title: Do environment variables overwrite the parameters passed to wandb.init()?
menu:
  support:
    identifier: ko-support-environment_variables_overwrite_parameters
tags:
- environment variables
toc_hide: true
type: docs
---

`wandb.init`에 전달된 인수는 환경 변수를 재정의합니다. 환경 변수가 설정되지 않았을 때 시스템 기본값 이외의 기본 디렉토리를 설정하려면 `wandb.init(dir=os.getenv("WANDB_DIR", my_default_override))`를 사용하세요.
