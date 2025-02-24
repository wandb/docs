---
title: Can we flag boolean variables as hyperparameters?
menu:
  support:
    identifier: ko-support-flag_boolean_variables_hyperparameters
tags:
- sweeps
toc_hide: true
type: docs
---

구성의 코맨드 섹션에서 `${args_no_boolean_flags}` 매크로를 사용하여 하이퍼파라미터를 boolean 플래그로 전달하세요. 이 매크로는 boolean 파라미터를 자동으로 플래그로 포함합니다. `param`이 `True`이면 코맨드는 `--param`을 받습니다. `param`이 `False`이면 플래그는 생략됩니다.
