
# wandb docker-run

**사용법**

`wandb docker-run [옵션] [DOCKER_RUN_ARGS]...`

**요약**

`docker run`을 감싸고 WANDB_API_KEY와 WANDB_DOCKER 환경
변수를 추가합니다.

이것은 시스템에 nvidia-docker 실행 파일이 있고 --runtime이 설정되지 않았다면 런타임을 nvidia로 설정할 것입니다.

자세한 내용은 `docker run --help`를 참조하세요.

**옵션**

| **옵션** | **설명** |
| :--- | :--- |