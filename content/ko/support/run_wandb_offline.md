---
title: Can I run wandb offline?
menu:
  support:
    identifier: ko-support-run_wandb_offline
tags:
- experiments
toc_hide: true
type: docs
---

만약 오프라인 장비에서 트레이닝이 수행되는 경우, 다음 단계를 따라 결과를 서버에 업로드하세요:

1. `WANDB_MODE=offline` 환경 변수를 설정하여 인터넷 연결 없이 로컬에 메트릭을 저장합니다.
2. 업로드할 준비가 되면 디렉토리에서 `wandb init` 을 실행하여 프로젝트 이름을 설정합니다.
3. `wandb sync YOUR_RUN_DIRECTORY` 를 사용하여 메트릭을 클라우드 서비스로 전송하고 호스팅된 웹 앱에서 결과에 엑세스합니다.

run 이 오프라인 상태인지 확인하려면 `wandb.init()` 을 실행한 후 `run.settings._offline` 또는 `run.settings.mode` 를 확인하십시오.
