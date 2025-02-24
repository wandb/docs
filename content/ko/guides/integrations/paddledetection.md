---
title: PaddleDetection
description: W&B를 PaddleDetection과 통합하는 방법.
menu:
  default:
    identifier: ko-guides-integrations-paddledetection
    parent: integrations
weight: 270
---

{{< cta-button colabLink="https://colab.research.google.com/drive/1ywdzcZKPmynih1GuGyCWB4Brf5Jj7xRY?usp=sharing" >}}

[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)은 [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) 기반의 엔드투엔드 오브젝트 감지 개발 키트입니다. 다양한 주류 오브젝트를 감지하고, 인스턴스를 분할하며, 네트워크 컴포넌트, 데이터 증강, 손실과 같은 구성 가능한 모듈을 사용하여 키포인트를 추적하고 감지합니다.

PaddleDetection은 이제 모든 트레이닝 및 검증 메트릭과 모델 체크포인트 및 해당 메타데이터를 기록하는 기본 제공 W&B 인테그레이션을 포함합니다.

PaddleDetection `WandbLogger`는 트레이닝 중에 트레이닝 및 평가 메트릭을 Weights & Biases에 기록하고 모델 체크포인트를 기록합니다.

[**W&B 블로그 게시물 읽기**](https://wandb.ai/manan-goel/PaddleDetectionYOLOX/reports/Object-Detection-with-PaddleDetection-and-W-B--VmlldzoyMDU4MjY0)는 `COCO2017` 데이터셋의 서브셋에서 YOLOX 모델을 PaddleDetection과 통합하는 방법을 보여줍니다.

## 가입하고 API 키 만들기

API 키는 사용자의 머신을 W&B에 인증합니다. 사용자 프로필에서 API 키를 생성할 수 있습니다.

{{% alert %}}
보다 간소화된 접근 방식을 위해 [https://wandb.ai/authorize](https://wandb.ai/authorize)로 직접 이동하여 API 키를 생성할 수 있습니다. 표시된 API 키를 복사하여 비밀번호 관리자와 같은 안전한 위치에 저장하십시오.
{{% /alert %}}

1. 오른쪽 상단 모서리에 있는 사용자 프로필 아이콘을 클릭합니다.
2. **User Settings**를 선택한 다음 **API Keys** 섹션으로 스크롤합니다.
3. **Reveal**을 클릭합니다. 표시된 API 키를 복사합니다. API 키를 숨기려면 페이지를 새로 고칩니다.

## `wandb` 라이브러리를 설치하고 로그인합니다.

`wandb` 라이브러리를 로컬에 설치하고 로그인하려면:

{{< tabpane text=true >}}
{{% tab header="커맨드라인" value="cli" %}}

1. API 키에 대해 `WANDB_API_KEY` [환경 변수]({{< relref path="/guides/models/track/environment-variables.md" lang="ko" >}})를 설정합니다.

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

{{% tab header="Python 노트북" value="python" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

## 트레이닝 스크립트에서 `WandbLogger`를 활성화합니다.

{{< tabpane text=true >}}
{{% tab header="커맨드라인" value="cli" %}}
[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/)에서 `train.py`에 대한 인수를 통해 wandb를 사용하려면:

* `--use_wandb` 플래그를 추가합니다.
* 첫 번째 wandb 인수는 `-o`로 시작해야 합니다(한 번만 전달하면 됨).
* 각 개별 wandb 인수는 접두사 `wandb-`를 포함해야 합니다. 예를 들어 [`wandb.init`]({{< relref path="/ref/python/init" lang="ko" >}})에 전달될 인수는 `wandb-` 접두사를 갖습니다.

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
`wandb` 키 아래의 config.yml 파일에 wandb 인수를 추가합니다.

```
wandb:
  project: MyProject
  entity: MyTeam
  save_dir: ./logs
```

`train.py` 파일을 실행하면 W&B 대시보드에 대한 링크가 생성됩니다.

{{< img src="/images/integrations/paddledetection_wb_dashboard.png" alt="A Weights & Biases Dashboard" >}}
{{% /tab %}}
{{< /tabpane >}}

## 피드백 또는 문제

Weights & Biases 인테그레이션에 대한 피드백이나 문제가 있는 경우 [PaddleDetection GitHub](https://github.com/PaddlePaddle/PaddleDetection)에서 문제를 열거나 <a href="mailto:support@wandb.com">support@wandb.com</a>으로 이메일을 보내주십시오.
