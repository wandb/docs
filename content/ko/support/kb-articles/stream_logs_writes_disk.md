---
title: How does wandb stream logs and writes to disk?
menu:
  support:
    identifier: ko-support-kb-articles-stream_logs_writes_disk
support:
- environment variables
toc_hide: true
type: docs
url: /ko/support/:filename
translationKey: stream_logs_writes_disk
---
W&B는 메모리에서 이벤트를 큐에 넣고 비동기적으로 디스크에 기록하여 오류를 관리하고 `WANDB_MODE=offline` 설정을 지원하여 로깅 후 동기화할 수 있도록 합니다.

터미널에서 로컬 run 디렉토리의 경로를 관찰합니다. 이 디렉토리에는 데이터 저장소 역할을 하는 `.wandb` 파일이 포함되어 있습니다. 이미지 로깅의 경우 W&B는 이미지를 클라우드 스토리지에 업로드하기 전에 `media/images` 하위 디렉토리에 저장합니다.
