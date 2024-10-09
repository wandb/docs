---
title: PaddleDetection
description: W&B를 PaddleDetection과 통합하는 방법.
slug: /guides/integrations/paddledetection
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)은 [PaddlePaddle](https://github.com/PaddlePaddle/Paddle)를 기반으로 한 엔드투엔드 오브젝트 검출 개발 키트입니다. 네트워크 컴포넌트, 데이터 증강 및 손실과 같은 구성 가능한 모듈을 사용하여 모듈형 디자인으로 다양한 주류 오브젝트 검출, 인스턴스 세그멘테이션, 추적 및 키포인트 검출 알고리즘을 구현합니다.

PaddleDetection에는 이제 W&B 인테그레이션이 내장되어 있어 모든 트레이닝 및 검증 메트릭, 모델 체크포인트 및 해당하는 메타데이터를 로그합니다.

## 예제 블로그 및 Colab

[**여기서 블로그를 읽으세요**](https://wandb.ai/manan-goel/PaddleDetectionYOLOX/reports/Object-Detection-with-PaddleDetection-and-W-B--VmlldzoyMDU4MjY0) 대하여 PaddleDetection을 사용하여 COCO2017 데이터셋의 서브셋에서 YOLOX 모델을 트레이닝하는 방법을 확인하세요. [**Google Colab**](https://colab.research.google.com/drive/1ywdzcZKPmynih1GuGyCWB4Brf5Jj7xRY?usp=sharing)이 포함되어 있으며, 해당하는 실시간 W&B 대시보드는 [**여기서**](https://wandb.ai/manan-goel/PaddleDetectionYOLOX/runs/2ry6i2x9?workspace=) 제공됩니다.

## PaddleDetection WandbLogger

PaddleDetection WandbLogger는 트레이닝 중에 Weights & Biases로 트레이닝 및 평가 메트릭을 로그하며 모델 체크포인트도 함께 기록합니다.

## Weights & Biases와 함께 PaddleDetection 사용하기

### W&B에 가입하고 로그인하기

무료 Weights & Biases 계정에 [**가입**](https://wandb.ai/site)한 후, wandb 라이브러리를 pip로 설치하세요. 로그인하려면 www.wandb.ai에 계정으로 로그인되어 있어야 합니다. 로그인 후 **API 키는** [**Authorize 페이지**](https://wandb.ai/authorize)**에서 찾을 수 있습니다.**

<Tabs
  defaultValue="cli"
  values={[
    {label: 'Command Line', value: 'cli'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="cli">

```shell
pip install wandb

wandb login
```
  </TabItem>
  <TabItem value="notebook">

```notebook
!pip install wandb

wandb.login()
```
  </TabItem>
</Tabs>

### 트레이닝 스크립트에서 WandbLogger 활성화하기

#### CLI 사용하기

[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/)의 `train.py`에 대한 인수로 wandb를 사용하려면:

* `--use_wandb` 플래그를 추가합니다
* 첫 번째 wandb 인수는 `-o`로 시작해야 합니다 (한 번만 전달하면 됨)
* 개별 wandb 인수는 `wandb-` 접두사를 포함해야 합니다. 예를 들어 [`wandb.init`](/ref/python/init)에 전달할 인수는 `wandb-` 접두사가 붙어야 합니다

```shell
python tools/train.py 
    -c config.yml \ 
    --use_wandb \
    -o \ 
    wandb-project=MyDetector \
    wandb-entity=MyTeam \
    wandb-save_dir=./logs
```

#### config.yml 파일 사용하기

구성 파일을 통해 wandb를 활성화할 수도 있습니다. 아래 wandb 헤더 아래 config.yml 파일에 wandb 인수를 다음과 같이 추가하세요:

```
wandb:
  project: MyProject
  entity: MyTeam
  save_dir: ./logs
```

Weights & Biases가 켜진 상태로 `train.py` 파일을 실행하면 W&B 대시보드로 이동할 수 있는 링크가 생성됩니다:

![A Weights & Biases Dashboard](/images/integrations/paddledetection_wb_dashboard.png)

## 피드백 또는 문제

Weights & Biases 인테그레이션에 대한 피드백이나 문제가 있는 경우, [PaddleDetection GitHub](https://github.com/PaddlePaddle/PaddleDetection)에서 이슈를 열거나 support@wandb.com으로 이메일을 보내주세요.