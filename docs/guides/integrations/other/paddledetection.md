---
description: How to integrate W&B with PaddleDetection.
slug: /guides/integrations/paddledetection
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# PaddleDetection

[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)은 [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) 기반의 엔드 투 엔드 오브젝트 디텍션 개발 키트입니다. 네트워크 구성요소, 데이터 증강 및 손실과 같은 설정 가능한 모듈을 사용한 모듈식 설계로 다양한 주류 오브젝트 디텍션, 인스턴스 세그멘테이션, 추적 및 키포인트 검출 알고리즘을 구현합니다.

PaddleDetection은 이제 학습 및 검증 메트릭, 모델 체크포인트 및 해당 메타데이터를 모두 로그하는 내장된 W&B 통합을 제공합니다.

## 예시 블로그 및 Colab

[**우리의 블로그를 읽어보세요**](https://wandb.ai/manan-goel/PaddleDetectionYOLOX/reports/Object-Detection-with-PaddleDetection-and-W-B--VmlldzoyMDU4MjY0) 여기서는 PaddleDetection을 사용하여 COCO2017 데이터세트의 서브세트에서 YOLOX 모델을 학습하는 방법을 볼 수 있습니다. 이는 [**Google Colab**](https://colab.research.google.com/drive/1ywdzcZKPmynih1GuGyCWB4Brf5Jj7xRY?usp=sharing)과 해당 실시간 W&B 대시보드가 [**여기**](https://wandb.ai/manan-goel/PaddleDetectionYOLOX/runs/2ry6i2x9?workspace=)에 있습니다.

## PaddleDetection WandbLogger

PaddleDetection WandbLogger는 학습 및 평가 메트릭을 Weights & Biases에 로그하고 학습 중에 모델 체크포인트를 로그합니다.

## Weights & Biases와 함께 PaddleDetection 사용하기

### W&B에 가입하고 로그인하기

[**가입하기**](https://wandb.ai/site)에서 무료 Weights & Biases 계정을 등록한 후, wandb 라이브러리를 pip로 설치하세요. 로그인하려면 www.wandb.ai에서 계정에 로그인해야 합니다. 로그인하고 나면 **API 키를** [**인증 페이지**](https://wandb.ai/authorize)**에서 찾을 수 있습니다.**

<Tabs
  defaultValue="cli"
  values={[
    {label: '명령 줄', value: 'cli'},
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

### 학습 스크립트에서 WandbLogger 활성화하기

#### CLI 사용하기

[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/)에서 `train.py`에 인수를 통해 wandb를 사용하려면:

* `--use_wandb` 플래그를 추가하세요
* 첫 번째 wandb 인수는 `-o`로 시작해야 합니다(이것은 한 번만 전달해야 합니다)
* 각 개별 wandb 인수는 `wandb-` 접두사를 포함해야 합니다. 예를 들어, [`wandb.init`](https://docs.wandb.ai/ref/python/init)에 전달할 인수는 `wandb-` 접두사가 붙습니다

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

## 피드백이나 문제점

Weights & Biases 통합에 대한 피드백이나 문제가 있으면 [PaddleDetection GitHub](https://github.com/PaddlePaddle/PaddleDetection)에 이슈를 열거나 support@wandb.com으로 이메일을 보내세요