---
title: Docker
description: W&B 와 Docker 를 통합하는 방법.
slug: /guides/integrations/docker
displayed_sidebar: default
---

## Docker 인테그레이션

W&B는 코드가 실행된 Docker 이미지에 대한 포인터를 저장하여, 이전 실험을 실행 당시의 정확한 환경으로 복원할 수 있는 기능을 제공합니다. wandb 라이브러리는 이 상태를 유지하기 위해 **WANDB_DOCKER** 환경 변수를 찾습니다. 우리는 이 상태를 자동으로 설정하는 몇 가지 헬퍼를 제공합니다.

### 로컬 개발

`wandb docker`는 docker 컨테이너를 시작하고, wandb 환경 변수를 전달하며, 코드를 마운트하고 wandb가 설치되었는지 확인하는 코맨드입니다. 기본적으로 이 코맨드는 TensorFlow, PyTorch, Keras, Jupyter가 설치된 docker 이미지를 사용합니다. 동일한 코맨드를 사용하여 자신의 docker 이미지를 시작할 수 있습니다: `wandb docker my/image:latest`. 이 코맨드는 현재 디렉토리를 컨테이너의 "/app" 디렉토리에 마운트하며, "--dir" 플래그로 이를 변경할 수 있습니다.

### 프로덕션

`wandb docker-run` 코맨드는 프로덕션 작업 부하를 위해 제공됩니다. 이는 `nvidia-docker`의 간편한 대체품으로 설계되었습니다. 이는 `docker run` 코맨드에 자격 증명과 **WANDB_DOCKER** 환경 변수를 추가하는 간단한 래퍼입니다. 만약 "--runtime" 플래그를 전달하지 않고, 머신에 `nvidia-docker`가 사용 가능하다면, 이 코맨드는 runtime을 nvidia로 설정되도록 보장합니다.

### Kubernetes

Kubernetes에서 트레이닝 작업 부하를 실행하고, k8s API가 pod에 노출된 경우(기본적으로 그렇습니다), wandb는 docker 이미지의 다이제스트를 위해 API를 쿼리하여 자동으로 **WANDB_DOCKER** 환경 변수를 설정합니다.

## 복원

**WANDB_DOCKER** 환경 변수를 사용하여 run을 계측한 경우, `wandb restore username/project:run_id`를 호출하면 코드를 복원하는 새 브랜치를 체크아웃한 후, 원래 사용된 코맨드로 사전 채워진 상태에서 트레이닝을 위해 사용된 정확한 docker 이미지를 실행합니다.