---
title: wandb 도커
menu:
  reference:
    identifier: ko-ref-cli-wandb-docker
---

**사용법**

`wandb docker [OPTIONS] [DOCKER_RUN_ARGS]... [DOCKER_IMAGE]`

**요약**

코드를 docker 컨테이너에서 실행합니다.

W&B docker를 사용하면 wandb가 설정된 상태로 docker 이미지에서 코드를 실행할 수 있습니다. WANDB_DOCKER와 WANDB_API_KEY 환경 변수( environment variables )를 컨테이너에 추가하고, 기본적으로 현재 디렉토리( directory )를 /app에 마운트합니다. 추가 ARG를 전달하면, 이미지 이름이 선언되기 전에 `docker run`에 추가됩니다. 이미지가 따로 지정되지 않은 경우, 기본 이미지를 선택해 사용합니다.

```sh
wandb docker -v /mnt/dataset:/app/data wandb docker gcr.io/kubeflow-images-public/tensorflow-1.12.0-notebook-cpu:v0.4.0 --jupyter
wandb docker wandb/deepo:keras-gpu --no-tty --cmd "python train.py --epochs=5"
```

기본적으로, entrypoint를 오버라이드하여 wandb가 이미 설치되어 있는지 확인하고 없을 경우 설치합니다. --jupyter 플래그를 전달하면 jupyter가 설치되어 있는지 확인하고, 포트 8888에서 jupyter lab을 시작합니다. 시스템에 nvidia-docker가 감지되면 nvidia 런타임을 사용합니다. wandb가 환경 변수만 기존 docker run 커맨드에 설정하도록 하고 싶다면, `wandb docker-run` 명령을 참고하세요.


**옵션**

| **옵션** | **설명** |
| :--- | :--- |
| `--nvidia / --no-nvidia` | nvidia 런타임 사용, nvidia-docker가 있을 경우 기본값은 nvidia |
| `--digest` | 이미지 digest를 출력하고 종료 |
| `--jupyter / --no-jupyter` | 컨테이너에서 jupyter lab 실행 |
| `--dir` | 컨테이너에 코드가 마운트될 디렉토리 지정 |
| `--no-dir` | 현재 디렉토리를 마운트하지 않음 |
| `--shell` | 컨테이너를 시작할 때 사용할 shell 지정 |
| `--port` | jupyter가 바인드될 호스트 포트 지정 |
| `--cmd` | 컨테이너에서 실행할 커맨드 지정 |
| `--no-tty` | tty 없이 커맨드 실행 |