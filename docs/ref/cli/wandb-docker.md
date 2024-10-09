# wandb docker

**사용법**

`wandb docker [옵션] [DOCKER_RUN_ARGS]... [DOCKER_IMAGE]`

**요약**

docker 컨테이너에서 당신의 코드를 실행하세요.

W&B docker를 사용하면 wandb가 구성된 docker 이미지에서 코드를 실행할 수 있습니다. WANDB_DOCKER와 WANDB_API_KEY 환경 변수를 컨테이너에 추가하고 기본적으로 현재 디렉토리를 /app에 마운트합니다. 추가적인 ARG을 전달할 수 있으며, 이는 이미지 이름이 선언되기 전에 `docker run`에 추가됩니다. 전달하지 않으면 기본 이미지를 선택합니다:

```sh
wandb docker -v /mnt/dataset:/app/data wandb docker gcr.io/kubeflow-
images-public/tensorflow-1.12.0-notebook-cpu:v0.4.0 --jupyter wandb docker
wandb/deepo:keras-gpu --no-tty --cmd "python train.py --epochs=5"
```

기본적으로 우리는 wandb의 존재 여부를 확인하고 설치되지 않은 경우 설치를 진행하기 위해 entrypoint를 덮어씁니다. --jupyter 플래그를 전달하면 우리는 jupyter가 설치되었는지 확인하고 포트 8888에서 jupyter lab을 시작합니다. 시스템에서 nvidia-docker가 감지되면 nvidia 런타임을 사용합니다. 기존 docker run 명령에 대해 wandb가 환경 변수를 설정하도록 하고 싶다면, wandb docker-run 명령을 참조하세요.

**옵션**

| **옵션** | **설명** |
| :--- | :--- |
| --nvidia / --no-nvidia | nvidia 런타임을 사용합니다. nvidia-docker가 있으면 기본값은 nvidia입니다. |
| --digest | 이미지 다이제스트를 출력하고 종료합니다. |
| --jupyter / --no-jupyter | 컨테이너에서 jupyter lab을 실행합니다. |
| --dir | 컨테이너에 마운트할 코드의 디렉토리입니다. |
| --no-dir | 현재 디렉토리를 마운트하지 않습니다. |
| --shell | 컨테이너를 시작할 셸입니다. |
| --port | jupyter를 바인드할 호스트 포트입니다. |
| --cmd | 컨테이너에서 실행할 명령입니다. |
| --no-tty | tty 없이 명령을 실행합니다. |