---
description: How to integrate W&B with Docker.
slug: /guides/integrations/docker
displayed_sidebar: default
---

# Docker

## Docker 통합

W&B는 코드가 실행된 Docker 이미지에 대한 포인터를 저장하여 이전 실험을 실행된 정확한 환경으로 복원할 수 있게 해줍니다. wandb 라이브러리는 이 상태를 유지하기 위해 **WANDB\_DOCKER** 환경 변수를 찾습니다. 자동으로 이 상태를 설정하는 몇 가지 도우미를 제공합니다.

### 로컬 개발

`wandb docker`는 docker 컨테이너를 시작하고, wandb 환경 변수를 전달하고, 코드를 마운트하며, wandb가 설치되어 있는지 확인하는 명령입니다. 기본적으로 이 명령은 TensorFlow, PyTorch, Keras, Jupyter가 설치된 docker 이미지를 사용합니다. 자신의 docker 이미지를 시작하려면 같은 명령을 사용하면 됩니다: `wandb docker my/image:latest`. 이 명령은 현재 디렉터리를 컨테이너의 "/app" 디렉터리에 마운트합니다. "--dir" 플래그를 사용하여 이를 변경할 수 있습니다.

### 프로덕션

프로덕션 작업 부하를 위해 `wandb docker-run` 명령이 제공됩니다. 이는 `nvidia-docker`의 대체제로 의도되었습니다. 이는 사용자의 자격증명과 **WANDB\_DOCKER** 환경 변수를 호출에 추가하는 `docker run` 명령의 간단한 래퍼입니다. "--runtime" 플래그를 전달하지 않고 기계에 `nvidia-docker`가 설치되어 있는 경우, 이는 런타임을 nvidia로 설정되도록 합니다.

### 쿠버네티스

쿠버네티스에서 학습 작업 부하를 실행하고 k8s API가 귀하의 포드에 노출되어 있으면(기본적으로 그런 경우입니다). wandb는 API를 쿼리하여 docker 이미지의 다이제스트를 자동으로 찾아 **WANDB\_DOCKER** 환경 변수를 설정합니다.

## 복원

실행이 **WANDB\_DOCKER** 환경 변수로 계측된 경우, `wandb restore username/project:run_id`를 호출하면 코드를 복원하는 새 브랜치를 체크아웃하고 학습에 사용된 정확한 docker 이미지를 원래 명령으로 미리 채워진 상태로 실행합니다.