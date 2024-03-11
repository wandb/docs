---
description: How to integrate W&B with Meta AI's MMF.
slug: /guides/integrations/mmf
displayed_sidebar: default
---

# MMF

`WandbLogger` 클래스는 [Meta AI의 MMF](https://github.com/facebookresearch/mmf) 라이브러리에서 Weights & Biases에 트레이닝/검증 메트릭, 시스템(GPU 및 CPU) 메트릭, 모델 체크포인트 및 설정 파라미터를 로그하는 기능을 활성화합니다.

### 현재 기능

현재 MMF의 `WandbLogger`에 의해 지원되는 기능은 다음과 같습니다:

* 트레이닝 & 검증 메트릭
* 시간에 따른 학습률
* 모델 체크포인트를 W&B Artifacts에 저장
* GPU 및 CPU 시스템 메트릭
* 트레이닝 설정 파라미터

### 설정 파라미터

MMF 설정에서 wandb 로깅을 활성화하고 사용자 정의할 수 있는 다음 옵션들이 있습니다:

```
training:
    wandb:
        enabled: true
        
        # 엔티티는 runs을 보내는 사용자 이름 또는 팀 이름입니다.
        # 기본적으로 사용자 계정에 run을 로그합니다.
        entity: null
        
        # wandb와 실험을 로깅할 때 사용될 프로젝트 이름
        project: mmf
        
        # wandb와 프로젝트 아래에서 실험을 로깅할 때 사용될
        # 실험/ run 이름. 기본 실험 이름은: ${training.experiment_name}
        name: ${training.experiment_name}
        
        # 모델 체크포인팅을 켜고 W&B Artifacts에 체크포인트를 저장
        log_model_checkpoint: true
        
        # wandb.init()에 전달하고자 하는 추가 인수 값들.
        # 사용 가능한 인수가 무엇인지 보려면 https://docs.wandb.ai/ref/python/init
        # 문서를 확인하세요. 예를 들면:
        # job_type: 'train'
        # tags: ['tag1', 'tag2']
        
env:
    # wandb 메타데이터가 저장될 디렉토리의 경로를 변경하려면
    # (기본값: env.log_dir):
    wandb_logdir: ${env:MMF_WANDB_LOGDIR,}
```