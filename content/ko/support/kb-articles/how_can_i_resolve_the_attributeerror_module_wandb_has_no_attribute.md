---
title: 'How can I fix an error like `AttributeError: module ''wandb'' has no attribute
  ...`?'
menu:
  support:
    identifier: ko-support-kb-articles-how_can_i_resolve_the_attributeerror_module_wandb_has_no_attribute
support:
- crashing and hanging runs
toc_hide: true
type: docs
url: /ko/support/:filename
translationKey: how_can_i_resolve_the_attributeerror_module_wandb_has_no_attribute
---
Python에서 `wandb`를 가져올 때 `AttributeError: module 'wandb' has no attribute 'init'` 또는 `AttributeError: module 'wandb' has no attribute 'login'`과 같은 오류가 발생하는 경우, `wandb`가 설치되지 않았거나 설치가 손상되었지만 현재 작업 디렉토리에 `wandb` 디렉토리가 존재하는 경우입니다. 이 오류를 해결하려면 `wandb`를 제거하고 해당 디렉토리를 삭제한 다음 `wandb`를 설치하세요.

```bash
pip uninstall wandb; rm -rI wandb; pip install wandb
```
