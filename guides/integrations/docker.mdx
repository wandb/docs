---
title: Docker
description: W&B를 Docker와 통합하는 방법.
menu:
  default:
    identifier: ko-guides-integrations-docker
    parent: integrations
weight: 80
---

## Docker 인테그레이션

W&B는 코드 가 실행된 Docker 이미지에 대한 포인터를 저장하여 이전의 실험 을 정확한 환경 으로 복원할 수 있도록 합니다. wandb 라이브러리 는 이 상태 를 유지하기 위해 **WANDB_DOCKER** 환경 변수 를 찾습니다. 이 상태 를 자동으로 설정하는 몇 가지 도우미를 제공합니다.

### 로컬 개발

`wandb docker` 는 docker 컨테이너 를 시작하고, wandb 환경 변수 를 전달하고, 코드 를 마운트하고, wandb가 설치되었는지 확인하는 코맨드 입니다. 기본적으로 이 코맨드 는 TensorFlow, PyTorch, Keras 및 Jupyter가 설치된 docker 이미지 를 사용합니다. 동일한 코맨드 를 사용하여 자신의 docker 이미지 를 시작할 수 있습니다: `wandb docker my/image:latest`. 이 코맨드 는 현재 디렉토리 를 컨테이너 의 "/app" 디렉토리 에 마운트합니다. "--dir" 플래그 를 사용하여 이를 변경할 수 있습니다.

### 프로덕션

`wandb docker-run` 코맨드 는 프로덕션 워크로드 를 위해 제공됩니다. `nvidia-docker` 를 대체할 수 있도록 만들어졌습니다. 이는 `docker run` 코맨드 에 대한 간단한 래퍼 로, 자격 증명 과 **WANDB_DOCKER** 환경 변수 를 호출에 추가합니다. "--runtime" 플래그 를 전달하지 않고 시스템에서 `nvidia-docker` 를 사용할 수 있는 경우 런타임 이 nvidia로 설정됩니다.

### Kubernetes

Kubernetes에서 트레이닝 워크로드 를 실행하고 k8s API가 Pod에 노출된 경우 (기본적으로 해당됨) wandb는 docker 이미지 의 다이제스트에 대해 API를 쿼리하고 **WANDB_DOCKER** 환경 변수 를 자동으로 설정합니다.

## 복원

**WANDB_DOCKER** 환경 변수 로 Run이 계측된 경우, `wandb restore username/project:run_id` 를 호출하면 코드 를 복원하는 새 분기를 체크아웃한 다음, 트레이닝 에 사용된 정확한 docker 이미지 를 원래 코맨드 로 미리 채워 시작합니다.
