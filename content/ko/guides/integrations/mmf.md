---
title: 'MMF


  '
description: W&B 를 Meta AI 의 MMF 와 연결하는 방법.
menu:
  default:
    identifier: ko-guides-integrations-mmf
    parent: integrations
weight: 220
---

`WandbLogger` 클래스는 [Meta AI의 MMF](https://github.com/facebookresearch/mmf) 라이브러리에서 W&B가 트레이닝/검증 메트릭, 시스템(GPU 및 CPU) 메트릭, 모델 체크포인트와 설정 파라미터를 로그로 남길 수 있게 해줍니다.

## 현재 지원되는 기능

MMF의 `WandbLogger`에서 현재 지원되는 기능은 다음과 같습니다:

* 트레이닝 & 검증 메트릭
* 시간 흐름에 따른 러닝 레이트
* 모델 체크포인트를 W&B Artifacts에 저장
* GPU 및 CPU 시스템 메트릭
* 트레이닝 설정 파라미터

## 설정 파라미터

MMF 설정에서 wandb 로깅을 활성화하고 커스터마이즈할 수 있는 옵션은 다음과 같습니다:

```
training:
    wandb:
        enabled: true
        
        # entity는 run을 업로드할 사용자명 또는 팀명입니다.
        # 기본값은 본인 사용자 계정에 run을 로깅합니다.
        entity: null
        
        # wandb로 실험을 기록할 때 사용할 프로젝트명입니다.
        project: mmf
        
        # 프로젝트 하위에서 실험을 기록할 때 사용할 실험 또는 run 이름입니다.
        # 기본 실험 이름은: ${training.experiment_name}
        name: ${training.experiment_name}
        
        # 모델 체크포인트 저장을 활성화하면 체크포인트가 W&B Artifacts에 기록됩니다.
        log_model_checkpoint: true
        
        # wandb.init()에 전달하고 싶은 추가 인수 값입니다. 예:
        # job_type: 'train'
        # tags: ['tag1', 'tag2']
        
env:
    # wandb 메타데이터가 저장되는 디렉토리 경로를 변경하려면 (기본값: env.log_dir):
    wandb_logdir: ${env:MMF_WANDB_LOGDIR,}
```