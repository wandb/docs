---
title: MMF
description: W&B를 Meta AI의 MMF와 통합하는 방법.
slug: /guides/integrations/mmf
displayed_sidebar: default
---

`WandbLogger` 클래스는 [Meta AI의 MMF](https://github.com/facebookresearch/mmf) 라이브러리에서 Weights & Biases가 트레이닝/검증 메트릭, 시스템 (GPU와 CPU) 메트릭, 모델 체크포인트 및 설정 파라미터를 로그할 수 있도록 합니다.

### 현재 기능

MMF의 `WandbLogger`에서 현재 지원되는 기능은 다음과 같습니다:

* 트레이닝 & 검증 메트릭
* 시간에 따른 학습률
* W&B Artifacts에 모델 체크포인트 저장
* GPU와 CPU 시스템 메트릭
* 트레이닝 설정 파라미터

### 설정 파라미터

wandb 로깅을 활성화하고 맞춤 설정하기 위해 MMF 설정에서 다음 옵션을 사용할 수 있습니다:

```
training:
    wandb:
        enabled: true
        
        # 엔터티는 실행을 보내고 있는 사용자명이나 팀명입니다.
        # 기본적으로 실행은 사용자 계정으로 로그됩니다.
        entity: null
        
        # wandb로 실험을 로그할 때 사용될 프로젝트 이름
        project: mmf
        
        # wandb와 함께 프로젝트에 실험을 로그할 때 사용될 실험/실행 이름.
        # 기본 실험 이름은: ${training.experiment_name}
        name: ${training.experiment_name}
        
        # 모델 체크포인트 기능을 활성화하여 W&B Artifacts에 체크포인트 저장
        log_model_checkpoint: true
        
        # wandb.init()에 전달하고 싶은 추가적인 인수 값들.
        # 사용할 수 있는 인수를 보려면 /ref/python/init 문서를 참조하세요. 예를 들면:
        # job_type: 'train'
        # tags: ['tag1', 'tag2']
        
env:
    # wandb 메타데이터가 저장될 디렉토리 경로를 변경하려면 
    # (기본값: env.log_dir):
    wandb_logdir: ${env:MMF_WANDB_LOGDIR,}
```