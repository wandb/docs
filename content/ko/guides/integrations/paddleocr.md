---
title: PaddleOCR
description: W&B 를 PaddleOCR 와 통합하는 방법
menu:
  default:
    identifier: ko-guides-integrations-paddleocr
    parent: integrations
weight: 280
---

[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)는 여러 언어를 지원하며 실제 적용이 가능한 선도적인 OCR 툴을 만드는 것을 목표로 하는 PaddlePaddle 기반 툴입니다. PaddleOCR는 OCR에 필요한 다양한 최신 알고리즘을 제공하고, 현장에서 바로 활용할 수 있는 솔루션도 지원합니다. 이제 PaddleOCR는 W&B와 인테그레이션되어, 트레이닝 및 평가 메트릭과 메타데이터가 포함된 모델 체크포인트를 간편하게 로그할 수 있습니다.

## 예시 블로그 & Colab

[PaddleOCR를 사용해 ICDAR2015 데이터셋으로 모델을 트레이닝하는 방법은 여기를 참고하세요.](https://wandb.ai/manan-goel/text_detection/reports/Train-and-Debug-Your-OCR-Models-with-PaddleOCR-and-W-B--VmlldzoyMDUwMDIw)  
이 가이드는 [Google Colab](https://colab.research.google.com/drive/1id2VTIQ5-M1TElAkzjzobUCdGeJeW-nV?usp=sharing) 예제까지 제공하며, 실제로 작동하는 W&B 대시보드는 [여기](https://wandb.ai/manan-goel/text_detection)에서 확인하실 수 있습니다. 이 블로그의 중국어 버전은 [여기](https://wandb.ai/wandb_fc/chinese/reports/W-B-OCR---VmlldzoyMDk1NzE4)에서 볼 수 있습니다.

## 회원가입 및 API 키 발급

API 키는 사용자의 머신이 W&B에 인증하는 데 필요합니다. 사용자 프로필에서 API 키를 발급받을 수 있습니다.

{{% alert %}}
가장 간편하게는 [W&B 인증 페이지](https://wandb.ai/authorize)에 바로 접속해 API 키를 생성할 수 있습니다. 표시된 API 키를 복사해서 비밀번호 관리자 등 안전한 곳에 보관하세요.
{{% /alert %}}

1. 화면 오른쪽 상단에서 사용자 프로필 아이콘을 클릭하세요.
1. **User Settings**를 선택한 후, **API Keys** 섹션까지 스크롤합니다.
1. **Reveal**을 클릭해 표시된 API 키를 복사하세요. 만약 API 키를 숨기고 싶다면 페이지를 새로고침하세요.

## `wandb` 라이브러리 설치 및 로그인

로컬 환경에 `wandb` 라이브러리를 설치하고 로그인하려면 아래를 참고하세요.

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [환경 변수]({{< relref path="/guides/models/track/environment-variables.md" lang="ko" >}})를 API 키 값으로 설정하세요.

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb` 라이브러리를 설치하고 로그인하세요.



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

{{% tab header="Python notebook" value="notebook" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

## wandb를 `config.yml` 파일에 추가

PaddleOCR는 여러 설정 변수들을 yaml 파일로 관리합니다. 설정 yaml 파일 마지막에 다음 내용을 추가하면 모든 트레이닝 및 검증 메트릭이 W&B 대시보드에 자동으로 로그되고, 모델 체크포인트도 함께 저장됩니다.

```python
Global:
    use_wandb: True
```

또, [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ko" >}})에 별도의 인수를 전달하고 싶다면 yaml 파일의 `wandb` 항목 아래 아래와 같이 작성하세요:

```
wandb:  
    project: CoolOCR  # (선택) wandb 프로젝트 이름
    entity: my_team   # (선택) wandb 팀이 있다면 팀 이름 입력
    name: MyOCRModel  # (선택) wandb run 이름
```

## `config.yml` 파일을 `train.py`에 전달

이 yaml 파일을 [트레이닝 스크립트](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/tools/train.py)에 인수로 넘겨 실행하세요.

```bash
python tools/train.py -c config.yml
```

`train.py`를 W&B가 활성화된 상태로 실행하면, 자동으로 W&B 대시보드 링크가 생성됩니다:

{{< img src="/images/integrations/paddleocr_wb_dashboard1.png" alt="PaddleOCR training dashboard" >}}

{{< img src="/images/integrations/paddleocr_wb_dashboard2.png" alt="PaddleOCR validation dashboard" >}}

{{< img src="/images/integrations/paddleocr_wb_dashboard3.png" alt="Text Detection Model dashboard" >}}

## 피드백 또는 이슈

W&B 인테그레이션 관련 피드백이나 문제가 있는 경우 [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)에 이슈를 남기거나, <a href="mailto:support@wandb.com">support@wandb.com</a>으로 문의해 주세요.