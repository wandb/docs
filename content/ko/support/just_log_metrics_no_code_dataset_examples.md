---
title: Can I just log metrics, no code or dataset examples?
menu:
  support:
    identifier: ko-support-just_log_metrics_no_code_dataset_examples
tags:
- administrator
- team management
- metrics
toc_hide: true
type: docs
---

기본적으로 W&B는 데이터셋 예제를 로깅하지 않습니다. 기본적으로 W&B는 코드 및 시스템 메트릭을 로깅합니다.

환경 변수를 사용하여 코드 로깅을 끄는 두 가지 방법이 있습니다:

1. `WANDB_DISABLE_CODE`를 `true`로 설정하여 모든 코드 추적을 끕니다. 이 작업을 수행하면 git SHA 및 diff patch 검색이 불가능해집니다.
2. `WANDB_IGNORE_GLOBS`를 `*.patch`로 설정하여 diff patch를 서버에 동기화하는 것을 중지하는 동시에 `wandb restore`를 사용하여 로컬에서 애플리케이션에 사용할 수 있도록 유지합니다.

관리자는 팀 설정에서 팀의 코드 저장을 끌 수도 있습니다:

1. `https://wandb.ai/<team>/settings`에서 팀 설정으로 이동합니다. 여기서 `<team>`은 팀 이름입니다.
2. 개인 정보 보호 섹션으로 스크롤합니다.
3. **기본적으로 코드 저장 활성화**를 전환합니다.
