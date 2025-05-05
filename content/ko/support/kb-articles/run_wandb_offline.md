---
title: Can I run wandb offline?
menu:
  support:
    identifier: ko-support-kb-articles-run_wandb_offline
support:
- experiments
toc_hide: true
type: docs
url: /ko/support/:filename
---

오프라인 머신에서 트레이닝이 발생하는 경우, 다음 단계를 통해 결과를 서버에 업로드하세요:

1. 인터넷 연결 없이 로컬에 메트릭을 저장하려면 환경 변수 `WANDB_MODE=offline`을 설정합니다.
2. 업로드할 준비가 되면 디렉토리에서 `wandb init`을 실행하여 프로젝트 이름을 설정합니다.
3. `wandb sync YOUR_RUN_DIRECTORY`를 사용하여 메트릭을 클라우드 서비스로 전송하고 호스팅된 웹 앱에서 결과에 엑세스합니다.

run이 오프라인인지 확인하려면 `wandb.init()`을 실행한 후 `run.settings._offline` 또는 `run.settings.mode`를 확인하세요.
