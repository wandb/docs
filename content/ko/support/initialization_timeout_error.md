---
title: How do I resolve a run initialization timeout error in wandb?
menu:
  support:
    identifier: ko-support-initialization_timeout_error
tags:
- connectivity
- crashing and hanging runs
toc_hide: true
type: docs
---

run 초기화 시간 초과 오류를 해결하려면 다음 단계를 따르세요.

- **초기화 재시도**: run을 다시 시작해 보세요.
- **네트워크 연결 확인**: 안정적인 인터넷 연결을 확인하세요.
- **wandb 버전 업데이트**: wandb의 최신 버전을 설치하세요.
- **시간 초과 설정 늘리기**: `WANDB_INIT_TIMEOUT` 환경 변수를 수정하세요.
  ```python
  import os
  os.environ['WANDB_INIT_TIMEOUT'] = '600'
  ```
- **디버깅 활성화**: 자세한 로그를 보려면 `WANDB_DEBUG=true` 및 `WANDB_CORE_DEBUG=true`로 설정하세요.
- **설정 확인**: API 키 와 project 설정이 올바른지 확인하세요.
- **로그 검토**: `debug.log`, `debug-internal.log`, `debug-core.log` 및 `output.log`에서 오류를 검사하세요.
