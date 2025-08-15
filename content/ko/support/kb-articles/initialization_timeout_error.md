---
title: wandb에서 run 초기화 타임아웃 오류를 어떻게 해결할 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-initialization_timeout_error
support:
- 연결성
- 크래시나 멈추는 run 처리하기
toc_hide: true
type: docs
url: /support/:filename
---

run 초기화 시간 초과 오류를 해결하려면 다음 단계를 따라 주세요:

- **초기화 재시도**: run 을 재시작해 보세요.
- **네트워크 연결 확인**: 안정적인 인터넷 연결 상태를 확인하세요.
- **wandb 버전 업데이트**: wandb 의 최신 버전을 설치하세요.
- **타임아웃 설정 증가**: `WANDB_INIT_TIMEOUT` 환경 변수 값을 변경하세요:
  ```python
  import os
  # 환경 변수로 타임아웃 시간을 600초로 설정
  os.environ['WANDB_INIT_TIMEOUT'] = '600'
  ```
- **디버깅 활성화**: `WANDB_DEBUG=true` 와 `WANDB_CORE_DEBUG=true` 를 설정해 상세 로그를 확인하세요.
- **설정 검증**: API 키와 프로젝트 설정이 올바른지 확인하세요.
- **로그 검토**: `debug.log`, `debug-internal.log`, `debug-core.log`, `output.log` 파일에서 오류를 확인하세요.