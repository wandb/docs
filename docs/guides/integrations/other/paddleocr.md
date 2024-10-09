---
title: PaddleOCR
description: W&B를 PaddleOCR에 통합하는 방법.
slug: /guides/integrations/paddleocr
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)는 PaddlePaddle에서 구현된 다국어, 놀랍고, 선도적이며, 실용적인 OCR 툴을 만들어 사용자들이 더 나은 모델을 트레이닝하고 실전에 적용할 수 있도록 돕는 것을 목표로 합니다. PaddleOCR은 OCR 관련 다양한 최첨단의 알고리즘을 지원하고, 산업용 솔루션을 개발했습니다. 이제 PaddleOCR은 Weights & Biases 인테그레이션을 통해 트레이닝과 평가 메트릭을 로그하고, 모델 체크포인트와 해당 메타데이터를 기록할 수 있습니다.

## 예제 블로그 & Colab

[**여기를 읽어보세요**](https://wandb.ai/manan-goel/text_detection/reports/Train-and-Debug-Your-OCR-Models-with-PaddleOCR-and-W-B--VmlldzoyMDUwMDIw) PaddleOCR을 사용하여 ICDAR2015 데이터셋에 모델을 트레이닝하는 방법을 알아볼 수 있습니다. 이는 [**Google Colab**](https://colab.research.google.com/drive/1id2VTIQ5-M1TElAkzjzobUCdGeJeW-nV?usp=sharing)과 함께 제공되며, 관련된 실시간 W&B 대시보드는 [**여기**](https://wandb.ai/manan-goel/text_detection)에서 확인할 수 있습니다. 중국어 버전의 블로그는 여기에서 볼 수 있습니다: [**W&B对您的OCR模型进行训练和调试**](https://wandb.ai/wandb_fc/chinese/reports/W-B-OCR---VmlldzoyMDk1NzE4)

## Weights & Biases와 함께 PaddleOCR 사용하기

### 1. wandb에 회원가입하고 로그인하기

[**회원가입**](https://wandb.ai/site)하여 무료 계정을 만든 후, 커맨드라인에서 Python 3 환경에 wandb 라이브러리를 설치하십시오. 로그인하려면 www.wandb.ai에서 계정에 로그인해야 하며, [**Authorize 페이지**](https://wandb.ai/authorize)에서 API 키를 찾을 수 있습니다.

<Tabs
  defaultValue="cli"
  values={[
    {label: 'Command Line', value: 'cli'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="cli">

```
pip install wandb

wandb login
```

  </TabItem>
  <TabItem value="notebook">

```python
!pip install wandb

wandb.login()
```

  </TabItem>
</Tabs>

### 2. `config.yml` 파일에 wandb 추가하기

PaddleOCR에서는 설정 변수들을 yaml 파일을 통해 제공해야 합니다. 설정 yaml 파일의 끝에 다음 스니펫을 추가하면 트레이닝과 검증 메트릭이 자동으로 W&B 대시보드에 로그되며, 모델 체크포인트도 함께 기록됩니다:

```python
Global:
    use_wandb: True
```

`wandb.init`에 전달할 추가적인, 선택적인 인수는 yaml 파일의 `wandb` 헤더 아래에 추가할 수 있습니다:

```
wandb:  
    project: CoolOCR  # (선택사항) 이는 wandb 프로젝트 이름입니다 
    entity: my_team   # (선택사항) wandb 팀을 사용하는 경우, 여기에서 팀 이름을 전달할 수 있습니다
    name: MyOCRModel  # (선택사항) 이는 wandb run의 이름입니다
```

### 3. `train.py`에 `config.yml` 파일 전달하기

그 후, yaml 파일을 PaddleOCR 리포지토리에 있는 [트레이닝 스크립트](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/tools/train.py)의 인수로 제공합니다.

```
python tools/train.py -c config.yml
```

Weights & Biases가 켜진 상태에서 `train.py` 파일을 실행하면 W&B 대시보드로 이동할 수 있는 링크가 생성됩니다:

![](/images/integrations/paddleocr_wb_dashboard1.png) ![](/images/integrations/paddleocr_wb_dashboard2.png)

![텍스트 감지 모델의 W&B 대시보드](/images/integrations/paddleocr_wb_dashboard3.png)

## 피드백 또는 문제사항?

Weights & Biases 인테그레이션에 대한 피드백이나 문제가 있다면 [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)에 이슈를 남기거나 support@wandb.com으로 이메일을 보내주세요.