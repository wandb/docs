---
title: How do I resolve permission errors when logging a run?
menu:
  support:
    identifier: ko-support-resolve_permission_errors_when_logging_wandb_entity
tags:
- runs
- security
toc_hide: true
type: docs
---

W&B 엔티티에 run을 로깅할 때 권한 오류를 해결하려면 다음 단계를 따르세요.

- **엔티티 및 프로젝트 이름 확인**: 코드에서 W&B 엔티티 및 프로젝트 이름의 철자와 대소문자가 올바른지 확인합니다.
- **권한 확인**: 필요한 권한이 관리자에 의해 부여되었는지 확인합니다.
- **로그인 자격 증명 확인**: 올바른 W&B 계정으로 로그인했는지 확인합니다. 다음 코드를 사용하여 run을 생성하여 테스트합니다.
  ```python
  import wandb

  run = wandb.init(entity="your_entity", project="your_project")
  run.log({'example_metric': 1})
  run.finish()
  ```
- **API 키 설정**: `WANDB_API_KEY` 환경 변수를 사용합니다.
  ```bash
  export WANDB_API_KEY='your_api_key'
  ```
- **호스트 정보 확인**: 사용자 지정 배포의 경우 호스트 URL을 설정합니다.
  ```bash
  wandb login --relogin --host=<host-url>
  export WANDB_BASE_URL=<host-url>
  ```