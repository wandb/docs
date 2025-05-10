---
title: What happens if internet connection is lost while I'm training a model?
menu:
  support:
    identifier: ko-support-kb-articles-internet_connection_lost_while_im_training_model
support:
- environment variables
toc_hide: true
type: docs
url: /ko/support/:filename
translationKey: internet_connection_lost_while_im_training_model
---
라이브러리가 인터넷에 연결할 수 없으면 재시도 루프에 들어가 네트워크가 복원될 때까지 메트릭 스트리밍을 계속 시도합니다. 이 시간 동안 프로그램은 계속 실행됩니다.

인터넷이 없는 시스템에서 실행하려면 `WANDB_MODE=offline`을 설정합니다. 이 설정은 메트릭을 로컬 하드 드라이브에 저장합니다. 나중에 `wandb sync DIRECTORY` 를 호출하여 데이터를 서버로 스트리밍합니다.
