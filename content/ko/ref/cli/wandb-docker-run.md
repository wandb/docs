---
title: wandb docker-run
menu:
  reference:
    identifier: ko-ref-cli-wandb-docker-run
---

**사용법**

`wandb docker-run [OPTIONS] [DOCKER_RUN_ARGS]...`

**요약**

`docker run`을 래핑하고 WANDB_API_KEY 및 WANDB_DOCKER 환경 변수를 추가합니다.

이것은 또한 시스템에 nvidia-docker 실행 파일이 있고 --runtime이 설정되지 않은 경우 런타임을 nvidia로 설정합니다.

자세한 내용은 `docker run --help`를 참조하십시오.

**옵션**

| **옵션** | **설명** |
| :--- | :--- |
