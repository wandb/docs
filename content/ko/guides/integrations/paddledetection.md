---
title: PaddleDetection
description: W&B를 PaddleDetection과 통합하는 방법
menu:
  default:
    identifier: ko-guides-integrations-paddledetection
    parent: integrations
weight: 270
---

{{< cta-button colabLink="https://colab.research.google.com/drive/1ywdzcZKPmynih1GuGyCWB4Brf5Jj7xRY?usp=sharing" >}}

[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)은 [PaddlePaddle](https://github.com/PaddlePaddle/Paddle)을 기반으로 한 엔드투엔드 오브젝트 디텍션 개발 키트입니다. PaddleDetection은 다양한 주류 오브젝트 탐지, 인스턴스 분할, 그리고 네트워크 컴포넌트, 데이터 증강, 손실 함수 등 구성 가능한 모듈을 통해 키포인트 추적과 탐지를 지원합니다.

PaddleDetection에는 이제 W&B 인테그레이션이 내장되어 있어, 트레이닝 및 검증 메트릭, 모델 체크포인트 그리고 관련 메타데이터까지 자동으로 로그할 수 있습니다.

PaddleDetection의 `WandbLogger`는 트레이닝 및 평가 메트릭을 W&B에 자동으로 기록하고, 동시에 트레이닝 중 생성되는 모델 체크포인트도 함께 관리합니다.

YOLOX 모델을 PaddleDetection에서 `COCO2017` 데이터셋의 서브셋을 이용해서 인테그레이션하는 방법에 대해서는 [W&B 블로그 포스팅](https://wandb.ai/manan-goel/PaddleDetectionYOLOX/reports/Object-Detection-with-PaddleDetection-and-W-B--VmlldzoyMDU4MjY0)을 참고해보세요.

## 회원가입 및 API 키 생성하기

API 키는 사용자의 머신을 W&B에 인증해주는 역할을 합니다. API 키는 사용자 프로필에서 생성할 수 있습니다.

{{% alert %}}
가장 쉽게 API 키를 생성하려면, [W&B 인증 페이지](https://wandb.ai/authorize)로 바로 이동하세요. 화면에 표시된 API 키를 복사해 비밀번호 관리 프로그램 등 안전한 곳에 저장해 두세요.
{{% /alert %}}

1. 오른쪽 상단에서 사용자 프로필 아이콘을 클릭하세요.
1. **User Settings**를 선택하고, **API Keys** 섹션까지 스크롤하세요.
1. **Reveal**을 클릭해 API 키를 복사하세요. API 키를 다시 숨기려면 페이지를 새로고침하세요.

## `wandb` 라이브러리 설치 및 로그인

`wandb` 라이브러리를 로컬 환경에 설치하고 로그인하는 방법입니다:

{{< tabpane text=true >}}
{{% tab header="커맨드라인" value="cli" %}}

1. [환경 변수]({{< relref path="/guides/models/track/environment-variables.md" lang="ko" >}}) `WANDB_API_KEY`에 본인의 API 키를 설정하세요.

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb` 라이브러리를 설치한 후 로그인하세요.

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

{{% tab header="Python 노트북" value="python" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

## 트레이닝 스크립트에서 `WandbLogger` 활성화하기

{{< tabpane text=true >}}
{{% tab header="커맨드라인" value="cli" %}}
[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)의 `train.py`에 다음과 같이 인수를 추가해 wandb를 사용할 수 있습니다:

* `--use_wandb` 플래그를 추가하세요.
* wandb 관련 인수를 가장 먼저 전달할 때 `-o`를 붙여주세요 (한 번만 지정하면 됩니다).
* 각 wandb 인수 앞에는 `"wandb-"` 접두사를 붙여야 합니다. 예를 들어 [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ko" >}})에 전달할 인수라면 `wandb-` 접두사를 사용하세요.

```shell
python tools/train.py 
    -c config.yml \ 
    --use_wandb \
    -o \ 
    wandb-project=MyDetector \
    wandb-entity=MyTeam \
    wandb-save_dir=./logs
```
{{% /tab %}}
{{% tab header="`config.yml`" value="config" %}}
config.yml 파일의 `wandb` 키 아래에 wandb 인수들을 추가하세요:

```
wandb:
  project: MyProject
  entity: MyTeam
  save_dir: ./logs
```

`train.py`를 실행하면 W&B 대시보드로 연결되는 링크가 자동으로 생성됩니다.

{{< img src="/images/integrations/paddledetection_wb_dashboard.png" alt="A W&B Dashboard" >}}
{{% /tab %}}
{{< /tabpane >}}

## 피드백 및 이슈

W&B 인테그레이션에 대한 피드백이나 이슈가 있으실 경우, [PaddleDetection GitHub](https://github.com/PaddlePaddle/PaddleDetection)에 이슈를 남기시거나 <a href="mailto:support@wandb.com">support@wandb.com</a>으로 메일을 보내주세요.