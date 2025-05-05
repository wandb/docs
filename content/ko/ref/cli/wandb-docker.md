---
title: wandb docker
menu:
  reference:
    identifier: ko-ref-cli-wandb-docker
---

**사용법**

`wandb docker [OPTIONS] [DOCKER_RUN_ARGS]... [DOCKER_IMAGE]`

**요약**

docker 컨테이너에서 코드를 실행합니다.

W&B docker를 사용하면 wandb가 구성되었는지 확인하면서 docker 이미지에서 코드를 실행할 수 있습니다. 이 명령어는 컨테이너에 `WANDB_DOCKER` 및 `WANDB_API_KEY` 환경 변수를 추가하고 기본적으로 현재 디렉토리를 /app에 마운트합니다. 이미지 이름이 선언되기 전에 `docker run`에 추가될 추가 인수를 전달할 수 있습니다. 이미지가 전달되지 않은 경우 기본 이미지를 선택합니다.

```sh wandb docker -v /mnt/dataset:/app/data wandb docker gcr.io/kubeflow-
images-public/tensorflow-1.12.0-notebook-cpu:v0.4.0 --jupyter wandb docker
wandb/deepo:keras-gpu --no-tty --cmd "python train.py --epochs=5" ```

기본적으로 wandb의 존재를 확인하고 없는 경우 설치하기 위해 진입점을 재정의합니다. `--jupyter` 플래그를 전달하면 jupyter가 설치되었는지 확인하고 8888 포트에서 jupyter lab을 시작합니다. 시스템에서 nvidia-docker를 감지하면 nvidia 런타임을 사용합니다. 기존 docker run 코맨드에 wandb가 환경 변수를 설정하도록 하려면 wandb docker-run 코맨드를 참조하세요.

**옵션**

| **옵션** | **설명** |
| :--- | :--- |
| `--nvidia / --no-nvidia` | nvidia 런타임을 사용합니다. nvidia-docker가 있는 경우 기본적으로 nvidia를 사용합니다. |
| `--digest` | 이미지 다이제스트를 출력하고 종료합니다. |
| `--jupyter / --no-jupyter` | 컨테이너에서 jupyter lab을 실행합니다. |
| `--dir` | 컨테이너에서 코드를 마운트할 디렉토리입니다. |
| `--no-dir` | 현재 디렉토리를 마운트하지 않습니다. |
| `--shell` | 컨테이너를 시작할 쉘입니다. |
| `--port` | jupyter를 바인딩할 호스트 포트입니다. |
| `--cmd` | 컨테이너에서 실행할 코맨드입니다. |
| `--no-tty` | tty 없이 코맨드를 실행합니다. |
