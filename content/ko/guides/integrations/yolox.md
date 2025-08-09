---
title: YOLOX
description: W&B 를 YOLOX 와 통합하는 방법
menu:
  default:
    identifier: ko-guides-integrations-yolox
    parent: integrations
weight: 490
---

[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)는 강력한 오브젝트 검출 성능을 가진 앵커 프리 YOLO 버전입니다. YOLOX와 W&B 인테그레이션을 통해 트레이닝, 검증, 시스템 관련 메트릭을 쉽게 로깅할 수 있고, 커맨드라인 인수 하나로 예측값도 인터랙티브하게 검증할 수 있습니다.

## 회원가입 및 API 키 생성

API 키는 사용자의 머신을 W&B에 인증하는 역할을 합니다. 사용자 프로필에서 API 키를 생성할 수 있습니다.

{{% alert %}}
더 간편한 방법으로, [W&B 인증 페이지](https://wandb.ai/authorize)에서 바로 API 키를 생성할 수도 있습니다. 표시된 API 키를 복사해서 비밀번호 관리자와 같은 안전한 위치에 저장하세요.
{{% /alert %}}

1. 오른쪽 위에 있는 프로필 아이콘을 클릭하세요.
1. **User Settings**를 선택한 다음, **API Keys** 섹션까지 스크롤하세요.
1. **Reveal**을 클릭한 후, 나타난 API 키를 복사하세요. 키를 숨기고 싶다면 페이지를 새로고침하면 됩니다.

## `wandb` 라이브러리 설치 및 로그인

로컬에 `wandb` 라이브러리를 설치하고 로그인하려면 다음을 따라 하세요.

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [환경 변수]({{< relref path="/guides/models/track/environment-variables.md" lang="ko" >}})에 본인의 API 키를 설정하세요.

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb` 라이브러리를 설치하고 로그인합니다.


    ```shell
    pip install wandb

    wandb login
    ```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```bash
pip install wandb
```
```python
import wandb
wandb.login()
```

{{% /tab %}}

{{% tab header="Python notebook" value="python" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

## 메트릭 로깅

W&B와 함께 로깅을 활성화하려면 `--logger wandb` 커맨드라인 인수를 사용하세요. 추가로 [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ko" >}})에서 사용하는 모든 인수도 전달할 수 있습니다. 각 인수 앞에 `wandb-`를 붙이면 됩니다.

`num_eval_imges`는 검증 세트 이미지와 예측값이 W&B 테이블에 로깅되어 모델 평가에 사용되는 개수를 조절합니다.

```shell
# wandb에 로그인
wandb login

# yolox 트레이닝 스크립트에 `wandb` logger 인수를 추가해서 실행
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

[YOLOX 트레이닝 및 검증 메트릭 대시보드 예시 →](https://wandb.ai/manan-goel/yolox-nano/runs/3pzfeom)

{{< img src="/images/integrations/yolox_example_dashboard.png" alt="YOLOX 트레이닝 대시보드" >}}

이 W&B 인테그레이션에 대해 궁금한 점이나 문제가 있으신가요? [YOLOX 저장소](https://github.com/Megvii-BaseDetection/YOLOX)에 이슈를 남겨주세요.