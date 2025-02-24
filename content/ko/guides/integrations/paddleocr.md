---
title: PaddleOCR
description: W&B를 PaddleOCR과 통합하는 방법.
menu:
  default:
    identifier: ko-guides-integrations-paddleocr
    parent: integrations
weight: 280
---

[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)은 다국어, 훌륭함, 선도적, 실용적인 OCR 툴을 만들어 사용자가 더 나은 모델을 트레이닝하고 PaddlePaddle에서 구현된 실제에 적용할 수 있도록 하는 것을 목표로 합니다. PaddleOCR은 OCR과 관련된 다양한 최첨단 알고리즘을 지원하고 산업 솔루션을 개발했습니다. 이제 PaddleOCR은 해당 메타데이터와 함께 모델 체크포인트와 함께 트레이닝 및 평가 메트릭을 로깅하기 위한 Weights & Biases 인테그레이션과 함께 제공됩니다.

## 블로그 및 Colab 예제

ICDAR2015 데이터셋에서 PaddleOCR로 모델을 트레이닝하는 방법을 보려면 [**여기**](https://wandb.ai/manan-goel/text_detection/reports/Train-and-Debug-Your-OCR-Models-with-PaddleOCR-and-W-B--VmlldzoyMDUwMDIw)를 읽어보세요. 이것은 [**Google Colab**](https://colab.research.google.com/drive/1id2VTIQ5-M1TElAkzjzobUCdGeJeW-nV?usp=sharing)과 함께 제공되며 해당 라이브 W&B 대시보드는 [**여기**](https://wandb.ai/manan-goel/text_detection)에서 사용할 수 있습니다. 이 블로그의 중국어 버전도 있습니다: [**W&B对您的OCR模型进行训练和调试**](https://wandb.ai/wandb_fc/chinese/reports/W-B-OCR---VmlldzoyMDk1NzE4)

## 가입 및 API 키 생성

API 키는 사용자의 머신을 W&B에 인증합니다. 사용자 프로필에서 API 키를 생성할 수 있습니다.

{{% alert %}}
보다 간소화된 접근 방식을 위해 [https://wandb.ai/authorize](https://wandb.ai/authorize)로 직접 이동하여 API 키를 생성할 수 있습니다. 표시된 API 키를 복사하여 비밀번호 관리자와 같은 안전한 위치에 저장합니다.
{{% /alert %}}

1. 오른쪽 상단 모서리에 있는 사용자 프로필 아이콘을 클릭합니다.
2. **User Settings**를 선택한 다음 **API Keys** 섹션으로 스크롤합니다.
3. **Reveal**을 클릭합니다. 표시된 API 키를 복사합니다. API 키를 숨기려면 페이지를 새로 고칩니다.

## `wandb` 라이브러리 설치 및 로그인

`wandb` 라이브러리를 로컬에 설치하고 로그인하려면:

{{< tabpane text=true >}}
{{% tab header="커맨드라인" value="cli" %}}

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

{{% tab header="Python 노트북" value="notebook" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

## `config.yml` 파일에 wandb 추가

PaddleOCR은 yaml 파일을 사용하여 구성 변수를 제공해야 합니다. 구성 yaml 파일의 끝에 다음 스니펫을 추가하면 모든 트레이닝 및 유효성 검사 메트릭이 모델 체크포인트와 함께 W&B 대시보드에 자동으로 로깅됩니다.

```python
Global:
    use_wandb: True
```

[`wandb.init`]({{< relref path="/ref/python/init" lang="ko" >}})에 전달하고 싶을 수 있는 추가적인 선택적 인수는 yaml 파일의 `wandb` 헤더 아래에 추가할 수도 있습니다.

```
wandb:  
    project: CoolOCR  # (선택 사항) wandb 프로젝트 이름입니다.
    entity: my_team   # (선택 사항) wandb 팀을 사용하는 경우 여기에 팀 이름을 전달할 수 있습니다.
    name: MyOCRModel  # (선택 사항) wandb run의 이름입니다.
```

## `config.yml` 파일을 `train.py`에 전달

그런 다음 yaml 파일은 PaddleOCR 리포지토리에서 사용할 수 있는 [트레이닝 스크립트](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/tools/train.py)에 인수로 제공됩니다.

```bash
python tools/train.py -c config.yml
```

Weights & Biases가 켜진 상태로 `train.py` 파일을 실행하면 W&B 대시보드로 이동하는 링크가 생성됩니다.

{{< img src="/images/integrations/paddleocr_wb_dashboard1.png" alt="" >}}

{{< img src="/images/integrations/paddleocr_wb_dashboard2.png" alt="" >}}

{{< img src="/images/integrations/paddleocr_wb_dashboard3.png" alt="텍스트 감지 모델용 W&B 대시보드" >}}

## 피드백 또는 문제

Weights & Biases 인테그레이션에 대한 피드백이나 문제가 있는 경우 [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)에서 문제를 열거나 <a href="mailto:support@wandb.com">support@wandb.com</a>으로 이메일을 보내주세요.
