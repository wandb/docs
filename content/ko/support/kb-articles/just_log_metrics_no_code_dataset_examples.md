---
title: 코드나 데이터셋 예시 없이 메트릭만 로그할 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-just_log_metrics_no_code_dataset_examples
support:
- 관리자
- 팀 관리
- 메트릭
toc_hide: true
type: docs
url: /support/:filename
---

기본적으로 W&B는 데이터셋 예시는 로그하지 않습니다. 기본 설정에서는 W&B가 코드와 시스템 메트릭을 로그합니다.

환경 변수로 코드 로그를 비활성화하는 방법은 두 가지가 있습니다:

1. `WANDB_DISABLE_CODE`를 `true`로 설정하면 모든 코드 추적이 비활성화됩니다. 이 설정은 git SHA와 diff 패치의 가져오기를 막습니다.
2. `WANDB_IGNORE_GLOBS`를 `*.patch`로 설정하면 diff 패치가 서버로 동기화되지 않으며, 로컬에서는 `wandb restore`로 적용할 수 있도록 유지됩니다.

관리자인 경우, 팀의 설정에서 코드 저장을 끌 수도 있습니다:

1. `https://wandb.ai/<team>/settings`에서 해당 팀의 설정 페이지로 이동하세요. 여기서 `<team>`은 여러분의 팀 이름입니다.
2. Privacy(개인정보 보호) 섹션까지 스크롤하세요.
3. **Enable code saving by default**(기본적으로 코드 저장 활성화) 토글을 변경하세요.