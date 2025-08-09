---
title: wandb docker-run
menu:
  reference:
    identifier: ko-ref-cli-wandb-docker-run
---

**사용법**

`wandb docker-run [OPTIONS] [DOCKER_RUN_ARGS]...`

**요약**

`docker run`을 감싸서 WANDB_API_KEY와 WANDB_DOCKER 환경 변수를 추가합니다.

시스템에 nvidia-docker 실행 파일이 있고 --runtime 옵션이 설정되지 않았다면 런타임도 nvidia로 자동 지정됩니다.

자세한 내용은 `docker run --help`를 참고하세요.


**옵션**

| **옵션** | **설명** |
| :--- | :--- |