---
title: YOLOX
description: W&B를 YOLOX와 통합하는 방법
menu:
  default:
    identifier: ko-guides-integrations-yolox
    parent: integrations
weight: 490
---

[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)는 오브젝트 검출 성능이 뛰어난 앵커 프리 버전의 YOLO입니다. YOLOX W&B 인테그레이션을 사용하면 트레이닝, 검증, 시스템 관련 메트릭 기록을 활성화할 수 있으며, 단일 커맨드라인 인수로 예측값을 대화형으로 검증할 수 있습니다.

## 가입 및 API 키 생성

API 키는 사용자의 머신을 W&B에 인증합니다. 사용자 프로필에서 API 키를 생성할 수 있습니다.

{{% alert %}}
보다 간소화된 접근 방식을 위해 [https://wandb.ai/authorize](https://wandb.ai/authorize) 로 바로 이동하여 API 키를 생성할 수 있습니다. 표시된 API 키를 복사하여 비밀번호 관리자와 같은 안전한 위치에 저장하십시오.
{{% /alert %}}

1. 오른쪽 상단 모서리에 있는 사용자 프로필 아이콘을 클릭합니다.
2. **User Settings** 를 선택한 다음 **API Keys** 섹션으로 스크롤합니다.
3. **Reveal** 을 클릭합니다. 표시된 API 키를 복사합니다. API 키를 숨기려면 페이지를 새로 고칩니다.

## `wandb` 라이브러리 설치 및 로그인

`wandb` 라이브러리를 로컬에 설치하고 로그인하는 방법:

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [환경 변수]({{< relref path="/guides/models/track/environment-variables.md" lang="ko" >}})를 API 키로 설정합니다.

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

2. `wandb` 라이브러리를 설치하고 로그인합니다.

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

## 메트릭 기록

`--logger wandb` 커맨드라인 인수를 사용하여 wandb를 통한 로깅을 활성화합니다. 선택적으로 [`wandb.init`]({{< relref path="/ref/python/init" lang="ko" >}})이 예상하는 모든 인수를 전달할 수도 있습니다. 각 인수 앞에 `wandb-`를 붙입니다.

`num_eval_imges`는 모델 평가를 위해 W&B 테이블에 기록되는 검증 세트 이미지 및 예측값의 수를 제어합니다.

```shell
# wandb에 로그인
wandb login

# `wandb` logger 인수로 yolox 트레이닝 스크립트를 호출합니다.
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

[YOLOX 트레이닝 및 검증 메트릭이 포함된 예제 대시보드 ->](https://wandb.ai/manan-goel/yolox-nano/runs/3pzfeom)

{{< img src="/images/integrations/yolox_example_dashboard.png" alt="" >}}

이 W&B 인테그레이션에 대한 질문이나 문제가 있습니까? [YOLOX repository](https://github.com/Megvii-BaseDetection/YOLOX)에서 이슈를 여세요.
