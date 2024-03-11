---
description: How to integrate W&B with PaddleDetection.
slug: /guides/integrations/paddledetection
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# PaddleDetection

[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)은 [PaddlePaddle](https://github.com/PaddlePaddle/Paddle)을 기반으로 한 엔드투엔드 오브젝트 검출 개발 키트입니다. 이는 네트워크 구성 요소, 데이터 증강 및 손실과 같은 구성 가능한 모듈을 사용하여 다양한 주류 오브젝트 검출, 인스턴스 세그멘테이션, 추적 및 키포인트 검출 알고리즘을 모듈식 설계로 구현합니다.

PaddleDetection은 이제 모든 트레이닝 및 검증 메트릭, 모델 체크포인트 및 해당 메타데이터를 로그하는 내장 W&B 인테그레이션과 함께 제공됩니다.

## 예제 블로그 및 Colab

[**여기에서 우리의 블로그를 읽어보세요**](https://wandb.ai/manan-goel/PaddleDetectionYOLOX/reports/Object-Detection-with-PaddleDetection-and-W-B--VmlldzoyMDU4MjY0) COCO2017 데이터셋의 서브셋에서 PaddleDetection으로 YOLOX 모델을 트레이닝하는 방법을 확인할 수 있습니다. 이는 [**Google Colab**](https://colab.research.google.com/drive/1ywdzcZKPmynih1GuGyCWB4Brf5Jj7xRY?usp=sharing)도 함께 제공되며 해당 실시간 W&B 대시보드는 [**여기**](https://wandb.ai/manan-goel/PaddleDetectionYOLOX/runs/2ry6i2x9?workspace=)에서 확인할 수 있습니다.

## PaddleDetection WandbLogger

PaddleDetection WandbLogger는 트레이닝 중에 트레이닝 및 평가 메트릭을 Weights & Biases에 로그하고 모델 체크포인트를 기록합니다.

## Weights & Biases와 함께 PaddleDetection 사용하기

### W&B에 가입하고 로그인하기

[**가입**](https://wandb.ai/site)하여 무료 Weights & Biases 계정을 만들고, wandb 라이브러리를 pip 설치하세요. 로그인하려면 www.wandb.ai에 계정으로 로그인해야 합니다. 로그인하면 **API 키를** [**인증 페이지에서 찾을 수 있습니다.**](https://wandb.ai/authorize)**.**

<Tabs
  defaultValue="cli"
  values={[
    {label: '커맨드라인', value: 'cli'},
    {label: '노트북', value: 'notebook'},
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

`train.py`에 인수를 통해 wandb를 사용하려면 [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/)에서:

* `--use_wandb` 플래그를 추가하세요
* 첫 번째 wandb 인수는 `-o`로 시작해야 합니다(이것은 한 번만 전달해야 합니다)
* 각각의 개별 wandb 인수는 `wandb-` 접두어를 포함해야 합니다. 예를 들어, [`wandb.init`](https://docs.wandb.ai/ref/python/init)로 전달될 모든 인수는 `wandb-` 접두어를 가져야 합니다

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

config 파일을 통해서도 wandb를 활성화할 수 있습니다. wandb 인수를 config.yml 파일의 wandb 헤더 아래에 다음과 같이 추가하세요:

```
wandb:
  project: MyProject
  entity: MyTeam
  save_dir: ./logs
```

Weights & Biases가 켜진 상태로 `train.py` 파일을 실행하면 W&B 대시보드로 이동하는 링크가 생성됩니다:

![Weights & Biases 대시보드](/images/integrations/paddledetection_wb_dashboard.png)

## 피드백 또는 문제

Weights & Biases 인테그레이션에 대한 피드백이나 문제가 있을 경우 [PaddleDetection GitHub](https://github.com/PaddlePaddle/PaddleDetection)에 이슈를 열거나 support@wandb.com으로 이메일을 보내주세요