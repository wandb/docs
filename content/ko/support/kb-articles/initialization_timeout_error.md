---
title: How do I resolve a run initialization timeout error in wandb?
menu:
  support:
    identifier: ko-support-kb-articles-initialization_timeout_error
support:
- connectivity
- crashing and hanging runs
toc_hide: true
type: docs
url: /ko/support/:filename
---

run 초기화 시간 초과 오류를 해결하려면 다음 단계를 따르세요.

- **초기화 재시도**: run 재시도를 시도합니다.
- **네트워크 연결 확인**: 안정적인 인터넷 연결을 확인합니다.
- **wandb 버전 업데이트**: 최신 버전의 wandb를 설치합니다.
- **시간 초과 설정 증가**: `WANDB_INIT_TIMEOUT` 환경 변수를 수정합니다.
  ```python
  import os
  os.environ['WANDB_INIT_TIMEOUT'] = '600'
  ```
- **디버깅 활성화**: 자세한 로그를 보려면 `WANDB_DEBUG=true` 및 `WANDB_CORE_DEBUG=true`를 설정합니다.
- **설정 확인**: API 키 와 프로젝트 설정이 올바른지 확인합니다.
- **로그 검토**: 오류가 있는지 `debug.log`, `debug-internal.log`, `debug-core.log` 및 `output.log`를 검사합니다.
