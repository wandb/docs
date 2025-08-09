---
title: 불리언 변수를 하이퍼파라미터로 표시할 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-flag_boolean_variables_hyperparameters
support:
- 스윕
toc_hide: true
type: docs
url: /support/:filename
---

설정의 코맨드 섹션에서 `${args_no_boolean_flags}` 매크로를 사용하여 하이퍼파라미터를 불리언 플래그로 전달할 수 있습니다. 이 매크로는 불리언 파라미터를 자동으로 플래그 형태로 포함시켜 줍니다. 만약 `param` 이 `True`라면, 코맨드는 `--param`을 받게 됩니다. `param` 이 `False`라면 해당 플래그는 생략됩니다.