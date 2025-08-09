---
title: 도커
description: W&B 를 Docker 와 통합하는 방법
menu:
  default:
    identifier: ko-guides-integrations-docker
    parent: integrations
weight: 80
---

## Docker 인테그레이션

W&B는 코드가 실행된 Docker 이미지의 포인터를 저장할 수 있어, 이전 실험을 실행된 환경 그대로 복원할 수 있습니다. wandb 라이브러리는 이 상태를 유지하기 위해 **WANDB_DOCKER** 환경 변수를 찾습니다. 이 상태를 자동으로 설정해주는 몇 가지 헬퍼들도 제공합니다.

### 로컬 개발

`wandb docker`는 docker 컨테이너를 시작하고, wandb 환경 변수를 전달하며, 사용자의 코드를 마운트하고, wandb가 설치되어 있는지 확인하는 코맨드입니다. 기본적으로 이 코맨드는 TensorFlow, PyTorch, Keras, Jupyter가 모두 설치된 docker 이미지를 사용합니다. 같은 코맨드로 본인의 docker 이미지를 사용할 수도 있습니다: `wandb docker my/image:latest`. 이 코맨드는 현재 디렉토리를 컨테이너의 "/app" 디렉토리에 마운트하며, "--dir" 플래그로 변경할 수 있습니다.

### 프로덕션

`wandb docker-run` 코맨드는 프로덕션 워크로드를 위해 제공됩니다. 이는 `nvidia-docker`의 대체용으로 사용할 수 있습니다. 이 코맨드는 `docker run`의 간단한 래퍼로, 사용자의 인증 정보와 **WANDB_DOCKER** 환경 변수를 자동으로 추가해줍니다. "--runtime" 플래그를 전달하지 않고, `nvidia-docker`가 머신에 설치되어 있는 경우에는, 런타임을 자동으로 nvidia로 설정해줍니다.

### Kubernetes

만약 Kubernetes에서 트레이닝 워크로드를 실행하고, k8s API가 pod에 노출되어 있다면\(기본적으로 그렇습니다\), wandb가 도커 이미지의 다이제스트 정보를 API로부터 가져와 **WANDB_DOCKER** 환경 변수를 자동으로 설정합니다.

## 복원(Restoring)

run이 **WANDB_DOCKER** 환경 변수와 함께 기록되었다면, `wandb restore username/project:run_id`를 실행하면 새로운 브랜치로 코드를 복원하고, 트레이닝에 사용되었던 동일한 docker 이미지를 실행해 원래의 코맨드가 미리 설정된 상태로 시작합니다.