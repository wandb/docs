---
title: 모델 트레이닝 중에 인터넷 연결이 끊기면 어떻게 되나요?
menu:
  support:
    identifier: ko-support-kb-articles-internet_connection_lost_while_im_training_model
support:
- 환경 변수
- 장애
toc_hide: true
type: docs
url: /support/:filename
---

라이브러리가 인터넷에 연결되지 않으면 재시도 루프에 들어가 네트워크가 복구될 때까지 메트릭 전송을 계속 시도합니다. 이 동안 프로그램은 계속 실행됩니다.

인터넷이 없는 머신에서 실행하려면 `WANDB_MODE=offline`을 설정하세요. 이 설정은 메트릭을 하드 드라이브에 로컬로 저장합니다. 이후, `wandb sync DIRECTORY`를 실행하여 데이터를 서버로 전송할 수 있습니다.