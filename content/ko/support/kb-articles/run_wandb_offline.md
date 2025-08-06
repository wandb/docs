---
title: wandb를 오프라인으로 실행할 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-run_wandb_offline
support:
- 실험
toc_hide: true
type: docs
url: /support/:filename
---

오프라인 머신에서 트레이닝이 진행되는 경우, 결과를 서버에 업로드하려면 다음 단계를 따르세요:

1. 환경 변수 `WANDB_MODE=offline` 을 설정하여 인터넷 연결 없이 메트릭을 로컬에 저장합니다.
2. 업로드할 준비가 되면, 디렉토리에서 `wandb init` 을 실행하여 프로젝트 이름을 설정하세요.
3. `wandb sync YOUR_RUN_DIRECTORY` 를 사용하여 메트릭을 클라우드 서비스로 전송하고, 호스팅된 웹 앱에서 결과에 엑세스할 수 있습니다.

run 이 오프라인 상태인지 확인하려면, `wandb.init()` 실행 후 `run.settings._offline` 또는 `run.settings.mode` 를 체크하세요.