---
description: How to integrate W&B with YOLOX.
slug: /guides/integrations/yolox
displayed_sidebar: default
---

# YOLOX

[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)는 오브젝트 검출에 강력한 성능을 보이는 앵커-프리 버전의 YOLO입니다. YOLOX는 단 하나의 커맨드라인 인수로 트레이닝, 검증 및 시스템 메트릭 로깅, 그리고 인터랙티브한 검증 예측값을 활성화할 수 있는 Weights & Biases 인테그레이션을 포함하고 있습니다.

## 시작하기

Weights & Biases와 함께 YOLOX를 사용하려면 먼저 [여기](https://wandb.ai/site)에서 Weights & Biases 계정을 등록해야 합니다.

그런 다음 `--logger wandb` 커맨드라인 인수를 사용하여 wandb와 함께 로깅을 활성화하세요. 선택적으로 [wandb.init](../../track/launch.md)이 기대하는 모든 인수를 전달할 수도 있으며, 각 인수의 시작 부분에 `wandb-`를 추가하면 됩니다.

**참고:** `num_eval_imges`는 검증 세트 이미지와 모델 평가를 위해 Weights & Biases 테이블에 로그되는 예측값의 수를 제어합니다.

```shell
# wandb 로그인
wandb login

# `wandb` 로거 인수를 사용하여 yolox 트레이닝 스크립트를 호출
python tools/train.py .... --logger wandb \
                wandb-project <프로젝트-이름> \
                wandb-entity <엔티티>
                wandb-name <run-이름> \
                wandb-id <run-id> \
                wandb-save_dir <저장-디렉토리> \
                wandb-num_eval_imges <이미지-수> \
                wandb-log_checkpoints <불리언>
```

## 예시

[YOLOX 트레이닝 및 검증 메트릭이 포함된 예시 대시보드 ->](https://wandb.ai/manan-goel/yolox-nano/runs/3pzfeom)

![](/images/integrations/yolox_example_dashboard.png)

이 Weights & Biases 인테그레이션에 대한 질문이나 문제가 있나요? [YOLOX github 저장소](https://github.com/Megvii-BaseDetection/YOLOX)에 이슈를 오픈하시면 확인 후 답변 드리겠습니다 :)