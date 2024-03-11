---
description: How to integrate W&B with Docker.
slug: /guides/integrations/docker
displayed_sidebar: default
---

# Docker

## Docker 인테그레이션

W&B는 코드가 실행된 Docker 이미지에 대한 포인터를 저장하여 이전 실험을 정확히 실행되었던 환경으로 복원할 수 있는 기능을 제공합니다. wandb 라이브러리는 이 상태를 유지하기 위해 **WANDB\_DOCKER** 환경 변수를 찾습니다. 이 상태를 자동으로 설정하는 몇 가지 헬퍼를 제공합니다.

### 로컬 개발

`wandb docker` 코맨드는 docker 컨테이너를 시작하고, wandb 환경 변수를 전달하며, 코드를 마운트하고, wandb가 설치되어 있는지 확인합니다. 기본적으로 이 코맨드는 TensorFlow, PyTorch, Keras, Jupyter가 설치된 docker 이미지를 사용합니다. `wandb docker my/image:latest`와 같은 코맨드로 자신의 docker 이미지를 시작할 수도 있습니다. 코맨드는 현재 디렉토리를 컨테이너의 "/app" 디렉토리에 마운트하며, "--dir" 플래그를 사용하여 이를 변경할 수 있습니다.

### 프로덕션

`wandb docker-run` 코맨드는 프로덕션 워크로드를 위해 제공됩니다. 이는 `nvidia-docker`의 대체품으로 의도되었습니다. 이는 자격증명과 **WANDB\_DOCKER** 환경 변수를 호출에 추가하는 `docker run` 코맨드의 간단한 래퍼입니다. "--runtime" 플래그를 전달하지 않고 기계에 `nvidia-docker`가 사용 가능한 경우, 런타임이 nvidia로 설정되도록 합니다.

### Kubernetes

Kubernetes에서 트레이닝 워크로드를 실행하고 k8s API가 파드에 노출된 경우(기본적으로 그렇습니다). wandb는 API를 쿼리하여 docker 이미지의 다이제스트를 자동으로 찾고 **WANDB\_DOCKER** 환경 변수를 자동으로 설정합니다.

## 복원

실행이 **WANDB\_DOCKER** 환경 변수로 계측된 경우, `wandb restore username/project:run_id`를 호출하면 코드를 복원하는 새로운 브랜치를 체크아웃하고 트레이닝에 사용된 정확한 docker 이미지를 원래 코맨드로 미리 채워진 상태로 실행합니다.