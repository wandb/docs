---
title: '`AttributeError: module ''wandb'' has no attribute ...` 와 같은 오류를 어떻게 해결할 수
  있나요?'
menu:
  support:
    identifier: ko-support-kb-articles-how_can_i_resolve_the_attributeerror_module_wandb_has_no_attribute
support:
- crashing 및 멈추는 Runs
toc_hide: true
type: docs
url: /support/:filename
---

Python에서 `wandb`를 import할 때 `AttributeError: module 'wandb' has no attribute 'init'` 또는 `AttributeError: module 'wandb' has no attribute 'login'` 과 같은 에러가 발생한다면, `wandb`가 설치되지 않았거나 설치가 손상된 상태에서 현재 작업 디렉토리에 `wandb` 디렉토리가 존재하기 때문입니다. 이 에러를 해결하려면 `wandb`를 먼저 삭제하고, 그 디렉토리를 지운 뒤, 다시 `wandb`를 설치하세요:

```bash
pip uninstall wandb; rm -rI wandb; pip install wandb
```