---
title: What does wandb.init do to my training process?
menu:
  support:
    identifier: ko-support-wandbinit_training_process
tags:
- environment variables
- experiments
toc_hide: true
type: docs
---

`wandb.init()`이 트레이닝 스크립트에서 실행되면 API 호출은 서버에서 run 오브젝트를 생성합니다. 새로운 프로세스가 시작되어 메트릭을 스트리밍하고 수집하여 기본 프로세스가 정상적으로 기능하도록 합니다. 스크립트는 로컬 파일에 쓰고, 별도의 프로세스는 시스템 메트릭을 포함한 데이터를 서버로 스트리밍합니다. 스트리밍을 끄려면 트레이닝 디렉토리에서 `wandb off`를 실행하거나 `WANDB_MODE` 환경 변수를 `offline`으로 설정하세요.
