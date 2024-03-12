
# wandb docker

**사용법**

`wandb docker [옵션] [DOCKER_RUN_ARGS]... [DOCKER_IMAGE]`

**요약**

코드를 docker 컨테이너에서 실행합니다.

W&B docker는 wandb가 구성된 상태로 docker 이미지에서 코드를 실행할 수 있게 합니다. WANDB_DOCKER 및 WANDB_API_KEY 환경 변수를 컨테이너에 추가하고 기본적으로 현재 디렉토리를 /app에 마운트합니다. 추가적인 인자를 전달할 수 있으며, 이는 이미지 이름이 선언되기 전에 `docker run`에 추가됩니다. 이미지를 전달하지 않으면 기본 이미지를 선택합니다:

```sh
wandb docker -v /mnt/dataset:/app/data wandb docker gcr.io/kubeflow-images-public/tensorflow-1.12.0-notebook-cpu:v0.4.0 --jupyter wandb docker wandb/deepo:keras-gpu --no-tty --cmd "python train.py --epochs=5"
```

기본적으로, entrypoint를 오버라이드하여 wandb의 존재 여부를 확인하고 없을 경우 설치합니다. --jupyter 플래그를 전달하면 jupyter가 설치되어 있음을 보장하고 8888 포트에서 jupyter lab을 시작합니다. 시스템에 nvidia-docker가 감지되면 nvidia 런타임을 사용합니다. 기존 docker run 명령어에 환경 변수만 설정하고 싶다면 wandb docker-run 명령어를 참조하세요.

**옵션**

| **옵션** | **설명** |
| :--- | :--- |
| --nvidia / --no-nvidia | nvidia 런타임 사용, nvidia-docker가 있으면 기본적으로 nvidia 선택 |
| --digest | 이미지 다이제스트를 출력하고 종료 |
| --jupyter / --no-jupyter | 컨테이너에서 jupyter lab 실행 |
| --dir | 컨테이너에서 코드를 마운트할 디렉토리 |
| --no-dir | 현재 디렉토리를 마운트하지 않음 |
| --shell | 컨테이너를 시작할 쉘 |
| --port | jupyter를 바인드할 호스트 포트 |
| --cmd | 컨테이너에서 실행할 명령어 |
| --no-tty | tty 없이 명령어 실행 |