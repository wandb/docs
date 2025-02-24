---
title: MMF
description: Meta AI의 MMF와 W&B를 통합하는 방법.
menu:
  default:
    identifier: ko-guides-integrations-mmf
    parent: integrations
weight: 220
---

[Meta AI의 MMF](https://github.com/facebookresearch/mmf) 라이브러리의 `WandbLogger` 클래스를 사용하면 Weights & Biases가 트레이닝/검증 메트릭, 시스템 (GPU 및 CPU) 메트릭, 모델 체크포인트 및 설정 파라미터를 기록할 수 있습니다.

## 현재 기능

MMF의 `WandbLogger`는 현재 다음과 같은 기능을 지원합니다.

* 트레이닝 및 검증 메트릭
* 시간에 따른 학습률
* W&B Artifacts에 모델 체크포인트 저장
* GPU 및 CPU 시스템 메트릭
* 트레이닝 설정 파라미터

## 설정 파라미터

wandb 로깅을 활성화하고 사용자 정의하기 위해 MMF 설정에서 다음 옵션을 사용할 수 있습니다.

```
training:
    wandb:
        enabled: true
        
        # 엔티티는 Runs을 보내는 사용자 이름 또는 Team 이름입니다.
        # 기본적으로 Run은 사용자 계정에 기록됩니다.
        entity: null
        
        # wandb로 실험을 기록하는 데 사용되는 Project 이름입니다.
        project: mmf
        
        # wandb로 Project에서 실험을 기록하는 데 사용되는 Experiment/Run 이름입니다.
        # 기본 실험 이름은 다음과 같습니다: ${training.experiment_name}
        name: ${training.experiment_name}
        
        # 모델 체크포인트 설정을 켜서 체크포인트를 W&B Artifacts에 저장합니다.
        log_model_checkpoint: true
        
        # wandb.init()에 전달할 추가 인수 값입니다.
        # 사용 가능한 인수를 보려면 /ref/python/init에서 설명서를 확인하세요 (예:
        # job_type: 'train'
        # tags: ['tag1', 'tag2']
        
env:
    # wandb 메타데이터가 저장될 디렉토리 경로를 변경하려면 다음을 수행하세요.
    # (기본값: env.log_dir):
    wandb_logdir: ${env:MMF_WANDB_LOGDIR,}
```
