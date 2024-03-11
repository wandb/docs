---
description: How to integrate W&B with PaddleOCR.
slug: /guides/integrations/paddleocr
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# PaddleOCR

[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)은 다국어 지원, 우수한 성능, 선도적이며 실용적인 OCR 툴을 만들어 사용자가 더 나은 모델을 트레이닝하고 PaddlePaddle에서 실제로 적용할 수 있도록 돕는 것을 목표로 합니다. PaddleOCR은 OCR과 관련된 다양한 최첨단 알고리즘을 지원하며, 산업 솔루션을 개발했습니다. PaddleOCR은 이제 트레이닝 및 평가 메트릭을 로깅하고 해당 메타데이터와 함께 모델 체크포인트를 Weights & Biases와 통합하여 제공합니다.

## 예제 블로그 & 콜랩

[**여기서 읽기**](https://wandb.ai/manan-goel/text\_detection/reports/Train-and-Debug-Your-OCR-Models-with-PaddleOCR-and-W-B--VmlldzoyMDUwMDIw) ICDAR2015 데이터셋에서 PaddleOCR로 모델을 트레이닝하는 방법을 확인해보세요. 이는 [**Google Colab**](https://colab.research.google.com/drive/1id2VTIQ5-M1TElAkzjzobUCdGeJeW-nV?usp=sharing)과 해당하는 실시간 W&B 대시보드가 [**여기**](https://wandb.ai/manan-goel/text\_detection)에 있습니다. 이 블로그의 중국어 버전은 여기에서 확인할 수 있습니다: [**W&B로 OCR 모델 트레이닝 및 디버깅하기**](https://wandb.ai/wandb\_fc/chinese/reports/W-B-OCR---VmlldzoyMDk1NzE4)

## Weights & Biases와 PaddleOCR 사용하기

### 1. wandb에 가입하고 로그인하기

[**가입하기**](https://wandb.ai/site)에서 무료 계정을 만든 후, Python 3 환경에서 wandb 라이브러리를 설치하세요. 로그인하려면 www.wandb.ai에 로그인한 상태여야 하며, **API 키는** [**인증 페이지**](https://wandb.ai/authorize)에서 찾을 수 있습니다.

<Tabs
  defaultValue="cli"
  values={[
    {label: '커맨드라인', value: 'cli'},
    {label: '노트북', value: 'notebook'},
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

PaddleOCR은 yaml 파일을 사용하여 설정 변수를 제공하도록 요구합니다. 설정 yaml 파일의 끝에 다음 스니펫을 추가하면 모든 트레이닝 및 검증 메트릭이 모델 체크포인트와 함께 W&B 대시보드에 자동으로 로깅됩니다:

```python
Global:
    use_wandb: True
```

yaml 파일의 `wandb` 헤더 아래에 [`wandb.init`](https://docs.wandb.ai/guides/track/launch)에 전달할 수 있는 추가적인, 선택적 인수도 추가할 수 있습니다:

```
wandb:  
    project: CoolOCR  # (선택사항) 이것은 wandb 프로젝트 이름입니다.
    entity: my_team   # (선택사항) wandb 팀을 사용하는 경우, 여기에 팀 이름을 전달할 수 있습니다.
    name: MyOCRModel  # (선택사항) 이것은 wandb run의 이름입니다.
```

### 3. `config.yml` 파일을 `train.py`에 전달하기

yaml 파일은 PaddleOCR 저장소에서 제공하는 [트레이닝 스크립트](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/tools/train.py)에 인수로 제공됩니다.

```
python tools/train.py -c config.yml
```

Weights & Biases가 켜진 상태로 `train.py` 파일을 실행하면 W&B 대시보드로 연결되는 링크가 생성됩니다:

![](/images/integrations/paddleocr_wb_dashboard1.png) ![](/images/integrations/paddleocr_wb_dashboard2.png)

![텍스트 탐지 모델을 위한 W&B 대시보드](/images/integrations/paddleocr_wb_dashboard3.png)

## 피드백이나 문제가 있나요?

Weights & Biases 통합에 대한 피드백이나 문제가 있으면 [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)에 이슈를 열거나 support@wandb.com으로 이메일을 보내세요.