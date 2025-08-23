---
title: 로그 중 run 을 기록할 때 권한 오류를 해결하려면 어떻게 해야 하나요?
menu:
  support:
    identifier: ko-support-kb-articles-resolve_permission_errors_when_logging_wandb_entity
support:
- run
- 보안
toc_hide: true
type: docs
url: /support/:filename
---

W&B entity 에 run 을 로그하는 중 권한 오류가 발생할 경우, 다음 단계들을 따라 해결할 수 있습니다:

- **entity 와 project 이름 확인**: 코드에서 사용한 W&B entity 와 project 이름의 철자 및 대소문자가 올바른지 확인하세요.
- **권한 확인**: 관리자가 필요한 권한을 부여했는지 확인하세요.
- **로그인 정보 확인**: 올바른 W&B 계정으로 로그인했는지 확인하세요. 아래 코드로 run 을 생성하여 테스트할 수 있습니다.
  ```python
  import wandb

  run = wandb.init(entity="your_entity", project="your_project")
  run.log({'example_metric': 1})
  run.finish()
  ```
- **API 키 설정**: `WANDB_API_KEY` 환경 변수를 사용하세요.
  ```bash
  export WANDB_API_KEY='your_api_key'
  ```
- **호스트 정보 확인**: 커스텀 배포 환경에서는 호스트 URL 을 설정해야 합니다.
  ```bash
  wandb login --relogin --host=<host-url>
  export WANDB_BASE_URL=<host-url>
  ```