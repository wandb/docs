---
description: How to integrate W&B with Meta AI's MMF.
slug: /guides/integrations/mmf
displayed_sidebar: default
---

# MMF

`WandbLogger` 클래스는 [Meta AI의 MMF](https://github.com/facebookresearch/mmf) 라이브러리에서 Weights & Biases가 학습/검증 메트릭, 시스템(GPU 및 CPU) 메트릭, 모델 체크포인트 및 구성 파라미터를 로그할 수 있게 해줍니다.

### 현재 기능

현재 MMF의 `WandbLogger`에서 지원하는 기능은 다음과 같습니다:

* 학습 및 검증 메트릭
* 시간에 따른 학습률
* 모델 체크포인트를 W&B 아티팩트에 저장
* GPU 및 CPU 시스템 메트릭
* 학습 구성 파라미터

### 구성 파라미터

MMF 구성에서 wandb 로깅을 활성화하고 사용자 정의하는 데 사용할 수 있는 옵션은 다음과 같습니다:

```
training:
    wandb:
        enabled: true
        
        # 엔티티는 실행을 보내는 사용자 이름 또는 팀 이름입니다.
        # 기본적으로 사용자 계정에 실행을 로그합니다.
        entity: null
        
        # wandb와 함께 실험을 로깅할 때 사용될 프로젝트 이름
        project: mmf
        
        # wandb와 프로젝트 아래에서 실험을 로깅할 때 사용될 실험/실행 이름
        # 기본 실험 이름은: ${training.experiment_name}
        name: ${training.experiment_name}
        
        # 모델 체크포인트를 켜고 W&B 아티팩트에 체크포인트를 저장
        log_model_checkpoint: true
        
        # wandb.init()에 전달하고 싶은 추가 인수 값입니다.
        # 사용할 수 있는 인수가 무엇인지 보려면 https://docs.wandb.ai/ref/python/init
        # 문서를 확인하세요. 예를 들어:
        # job_type: 'train'
        # tags: ['tag1', 'tag2']
        
env:
    # wandb 메타데이터가 저장될 디렉터리 경로를 변경하려면
    # (기본값: env.log_dir):
    wandb_logdir: ${env:MMF_WANDB_LOGDIR,}
```