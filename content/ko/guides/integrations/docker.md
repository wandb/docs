---
title: Docker
description: W&B를 Docker와 통합하는 방법.
menu:
  default:
    identifier: ko-guides-integrations-docker
    parent: integrations
weight: 80
---

## Docker Integration

W&B는 코드 실행에 사용된 Docker 이미지에 대한 포인터를 저장할 수 있어 이전 실험을 정확한 환경으로 복원할 수 있습니다. wandb 라이브러리는 이 상태를 유지하기 위해 **WANDB_DOCKER** 환경 변수를 찾습니다. 이 상태를 자동으로 설정하는 몇 가지 도우미를 제공합니다.

### Local Development

`wandb docker`는 Docker 컨테이너를 시작하고, wandb 환경 변수를 전달하고, 코드를 마운트하고, wandb가 설치되었는지 확인하는 코맨드입니다. 기본적으로 이 코맨드는 TensorFlow, PyTorch, Keras 및 Jupyter가 설치된 Docker 이미지를 사용합니다. 동일한 코맨드를 사용하여 자신의 Docker 이미지를 시작할 수 있습니다: `wandb docker my/image:latest`. 이 코맨드는 현재 디렉토리를 컨테이너의 "/app" 디렉토리에 마운트합니다. "--dir" 플래그를 사용하여 이를 변경할 수 있습니다.

### Production

`wandb docker-run` 코맨드는 프로덕션 워크로드를 위해 제공됩니다. 이는 `nvidia-docker`를 대체하기 위한 것입니다. 이는 자격 증명과 **WANDB_DOCKER** 환경 변수를 호출에 추가하는 `docker run` 코맨드에 대한 간단한 래퍼입니다. "--runtime" 플래그를 전달하지 않고 시스템에서 `nvidia-docker`를 사용할 수 있는 경우 런타임이 nvidia로 설정되었는지도 확인합니다.

### Kubernetes

Kubernetes에서 트레이닝 워크로드를 실행하고 k8s API가 pod에 노출된 경우 (기본적으로 해당됨) wandb는 Docker 이미지의 다이제스트에 대해 API를 쿼리하고 **WANDB_DOCKER** 환경 변수를 자동으로 설정합니다.

## Restoring

run이 **WANDB_DOCKER** 환경 변수로 계측된 경우 `wandb restore username/project:run_id`를 호출하면 코드를 복원하는 새 분기가 체크아웃되고 트레이닝에 사용된 정확한 Docker 이미지가 원래 코맨드로 미리 채워져 시작됩니다.
