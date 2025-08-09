---
title: wandb 는 로그를 어떻게 스트리밍하고 디스크에 기록하나요?
menu:
  support:
    identifier: ko-support-kb-articles-stream_logs_writes_disk
support:
- 환경 변수
toc_hide: true
type: docs
url: /support/:filename
---

W&B 는 이벤트를 메모리에 큐로 저장하고 비동기적으로 디스크에 기록하여 장애를 관리하고 `WANDB_MODE=offline` 설정을 지원합니다. 덕분에 로그를 남긴 후 동기화할 수 있습니다.

터미널에서 로컬 run 디렉토리의 경로를 확인할 수 있습니다. 이 디렉토리에는 .wandb 파일이 포함되어 있으며, 이는 데이터 저장소 역할을 합니다. 이미지 로그의 경우 W&B 는 이미지를 클라우드 저장소에 업로드하기 전에 media/images 하위 디렉토리에 저장합니다.