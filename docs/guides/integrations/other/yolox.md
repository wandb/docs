---
title: YOLOX
description: YOLOX와 W&B를 통합하는 방법.
slug: /guides/integrations/yolox
displayed_sidebar: default
---

[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)는 오브젝트 검출에서 뛰어난 성능을 보이는 앵커리스 버전의 YOLO입니다. YOLOX에는 트레이닝, 검증 및 시스템 메트릭의 로그를 켜고 대화형 검증 예측값을 단 1개의 커맨드라인 인수로 활성화할 수 있는 Weights & Biases 인테그레이션이 포함되어 있습니다.

## 시작하기

YOLOX를 Weights & Biases와 함께 사용하려면 먼저 [여기](https://wandb.ai/site)에서 Weights & Biases 계정을 가입해야 합니다.

그런 다음 `--logger wandb` 커맨드라인 인수를 사용하여 wandb로 로그를 켜면 됩니다. 선택적으로 [wandb.init](../../track/launch.md)이 기대하는 모든 인수를 전달할 수 있으며, 모든 인수의 시작 부분에 `wandb-`를 추가하면 됩니다.

**참고:** `num_eval_imges`는 모델 평가를 위해 Weights & Biases Tables에 로그될 검증 세트 이미지와 예측값의 수를 제어합니다.

```shell
# wandb에 로그인
wandb login

# `wandb` 로거 인수와 함께 yolox 트레이닝 스크립트를 호출
python tools/train.py .... --logger wandb \
                wandb-project <project-name> \
                wandb-entity <entity>
                wandb-name <run-name> \
                wandb-id <run-id> \
                wandb-save_dir <save-dir> \
                wandb-num_eval_imges <num-images> \
                wandb-log_checkpoints <bool>
```

## 예시

[YOLOX 트레이닝 및 검증 메트릭 대시보드 예시 ->](https://wandb.ai/manan-goel/yolox-nano/runs/3pzfeom)

![](/images/integrations/yolox_example_dashboard.png)

Weights & Biases 인테그레이션에 대한 질문이나 문제가 있습니까? [YOLOX github 리포지토리](https://github.com/Megvii-BaseDetection/YOLOX)에 이슈를 열어주시면 저희가 답변 드리겠습니다 :)