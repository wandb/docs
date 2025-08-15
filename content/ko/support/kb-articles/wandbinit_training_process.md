---
title: wandb.init 은 내 트레이닝 프로세스에 어떤 영향을 미치나요?
menu:
  support:
    identifier: ko-support-kb-articles-wandbinit_training_process
support:
- 환경 변수
- 실험
toc_hide: true
type: docs
url: /support/:filename
---

트레이닝 스크립트에서 `wandb.init()`가 실행되면, API 호출을 통해 서버에 run 오브젝트가 생성됩니다. 새로운 프로세스가 시작되어 메트릭을 스트리밍 및 수집하며, 기본 프로세스는 정상적으로 동작할 수 있습니다. 스크립트는 로컬 파일에 기록하고, 별도의 프로세스가 데이터와 시스템 메트릭을 서버로 스트리밍합니다. 스트리밍을 끄려면 트레이닝 디렉토리에서 `wandb off`를 실행하거나 `WANDB_MODE` 환경 변수를 `offline`으로 설정하세요.