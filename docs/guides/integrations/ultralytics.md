---
title: Ultralytics
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/01_train_val.ipynb"></CTAButtons>

[Ultralytics](https://github.com/ultralytics/ultralytics)는 이미지 분류, 오브젝트 검출, 이미지 세그멘테이션, 포즈 추정과 같은 최첨단의 컴퓨터 비전 모델을 위한 홈입니다. 여기에는 YOLO 시리즈의 최신 실시간 오브젝트 검출 모델인 [YOLOv8](https://docs.ultralytics.com/models/yolov8/)뿐만 아니라 [SAM (Segment Anything Model)](https://docs.ultralytics.com/models/sam/#introduction-to-sam-the-segment-anything-model), [RT-DETR](https://docs.ultralytics.com/models/rtdetr/), [YOLO-NAS](https://docs.ultralytics.com/models/yolo-nas/) 등 강력한 컴퓨터 비전 모델도 포함되어 있습니다. 이 모델들의 구현을 제공할 뿐만 아니라, Ultralytics는 이 모델들을 사용하여 쉽게 활용할 수 있는 엔드투엔드 워크플로우를 제공합니다.

## 시작하기

먼저, `ultralytics`와 `wandb`를 설치해야 합니다.

<Tabs
  defaultValue="script"
  values={[
    {label: '커맨드라인', value: 'script'},
    {label: '노트북', value: 'notebook'},
  ]}>
  <TabItem value="script">

```shell
pip install --upgrade ultralytics==8.0.238 wandb

# 또는
# conda install ultralytics
```

  </TabItem>
  <TabItem value="notebook">

```python
!pip install --upgrade ultralytics==8.0.238 wandb
```

  </TabItem>
</Tabs>

**안내:** 현재의 인테그레이션은 `ultralyticsv8.0.238`까지 테스트되었습니다. 문제 발생 시 https://github.com/wandb/wandb/issues에 `yolov8` 태그와 함께 보고해 주세요.

## Experiment 추적 및 검증 결과 시각화

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/01_train_val.ipynb"></CTAButtons>

이 섹션에서는 [Ultralytics](https://docs.ultralytics.com/modes/predict/) 모델을 사용하여 트레이닝, 파인튜닝, 검증을 수행하고, experiment 추적, 모델 체크포인트 실행 및 모델 성능 시각화를 [W&B](https://wandb.ai/site)를 사용하여 수행하는 일반적인 워크플로우를 보여줍니다.

코드를 Google Colab에서 시도해 보세요: [Open In Colab](http://wandb.me/ultralytics-train)

아래 리포트에서 인테그레이션에 대한 정보를 더 확인하세요: [Supercharging Ultralytics with W&B](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)

Ultralytics와 W&B 인테그레이션을 사용하기 위해서는 `wandb.integration.ultralytics.add_wandb_callback` 함수를 가져와야 합니다.

```python
import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics import YOLO
```

다음으로, 원하는 `YOLO` 모델을 초기화하고, 모델과 추론을 수행하기 전에 `add_wandb_callback`을 호출합니다. 이로 인해 트레이닝, 파인튜닝, 검증 또는 추론을 수행할 때 자동으로 experiment 로그와 예측 결과와 관련된 이미지가 [컴퓨터 비전 작업을 위한 인터랙티브 오버레이](../track/log/media#image-overlays-in-tables)를 사용하여 W&B에 기록되고, 추가적인 인사이트가 [`wandb.Table`](../tables/intro.md)에 표시됩니다.

```python
# YOLO 모델 초기화
model = YOLO("yolov8n.pt")

# Ultralytics를 위한 W&B 콜백 추가
add_wandb_callback(model, enable_model_checkpointing=True)

# 모델을 트레이닝/파인튜닝하세요
# 각 에포크 말에는 검증 배치에 대한 예측값이
# 컴퓨터 비전 작업을 위한 유익한 인터랙티브 오버레이와 함께
# W&B 테이블에 로그됩니다
model.train(project="ultralytics", data="coco128.yaml", epochs=5, imgsz=640)

# W&B run 종료
wandb.finish()
```

다음은 Ultralytics의 트레이닝 또는 파인튜닝 워크플로우에서 W&B를 사용하여 추적된 experiment이 어떻게 보이는지에 대한 예시입니다:

<blockquote class="imgur-embed-pub" lang="en" data-id="a/TB76U9O"  ><a href="//imgur.com/a/TB76U9O">YOLO 파인튜닝 Experiments</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

다음은 [W&B Table](../tables/intro.md)를 사용하여 에포크 별 검증 결과가 시각화된 모습입니다:

<blockquote class="imgur-embed-pub" lang="en" data-id="a/kU5h7W4"  ><a href="//imgur.com/a/kU5h7W4">WandB 검증 시각화 테이블</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

## 예측 결과 시각화

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/00_inference.ipynb"></CTAButtons>

이 섹션에서는 [Ultralytics](https://docs.ultralytics.com/modes/predict/) 모델을 사용하여 추론을 수행하고, [W&B](https://wandb.ai/site)를 사용하여 결과를 시각화하는 일반적인 워크플로우를 보여줍니다.

코드를 Google Colab에서 시도해 보세요: [Open in Colab](http://wandb.me/ultralytics-inference).

아래 리포트에서 인테그레이션에 대한 정보를 더 확인하세요: [Supercharging Ultralytics with W&B](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)

Ultralytics와 W&B 인테그레이션을 사용하기 위해서는 `wandb.integration.ultralytics.add_wandb_callback` 함수를 가져와야 합니다.

```python
import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics.engine.model import YOLO
```

이제, 인테그레이션을 테스트할 몇 가지 이미지를 다운로드해 보겠습니다. 사용자의 이미지, 비디오 또는 카메라 소스를 활용할 수 있습니다. 추론 소스에 대한 자세한 정보는 [공식 문서](https://docs.ultralytics.com/modes/predict/)를 참조하세요.

```python
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img1.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img2.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img4.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img5.png
```

그다음, `wandb.init`을 사용하여 W&B [run](../runs/intro.md)을 초기화합니다.

```python
# W&B run 초기화
wandb.init(project="ultralytics", job_type="inference")
```

다음, 원하는 `YOLO` 모델을 초기화하고, 모델과 추론을 수행하기 전에 `add_wandb_callback`을 호출합니다. 이로 인해 추론을 수행할 때 자동으로 이미지가 [컴퓨터 비전 작업을 위한 인터랙티브 오버레이](../track/log/media#image-overlays-in-tables)와 함께 W&B에 기록되고, 추가적인 인사이트가 [`wandb.Table`](../tables/intro.md)에 표시됩니다.

```python
# YOLO 모델 초기화
model = YOLO("yolov8n.pt")

# Ultralytics를 위한 W&B 콜백 추가
add_wandb_callback(model, enable_model_checkpointing=True)

# 예측 수행, 이는 자동으로 W&B 테이블에
# 바운딩 박스와 세그멘테이션 마스크에
# 대한 인터랙티브 오버레이와 함께 로그됩니다
model(["./assets/img1.jpeg", "./assets/img3.png", "./assets/img4.jpeg", "./assets/img5.jpeg"])

# W&B run 종료
wandb.finish()
```

안내: 트레이닝이나 파인튜닝 워크플로우의 경우 `wandb.init()`을 통해 run을 명시적으로 초기화할 필요가 없습니다. 그러나 코드가 오직 예측만 포함하는 경우에는 run을 명시적으로 생성해야 합니다.

다음은 인터랙티브 박스 오버레이의 예시입니다:

<blockquote class="imgur-embed-pub" lang="en" data-id="a/UTSiufs"  ><a href="//imgur.com/a/UTSiufs">WandB 이미지 오버레이</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

W&B 이미지 오버레이에 대한 정보를 더 확인할 수 있습니다 [여기에서](../track/log/media.md#image-overlays).

## 추가 리소스

* [Supercharging Ultralytics with Weights & Biases](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)
* [YOLOv8을 사용한 오브젝트 검출: 엔드투엔드 워크플로우](https://wandb.ai/reviewco/object-detection-bdd/reports/Object-Detection-using-YOLOv8-An-End-to-End-Workflow--Vmlldzo1NTAyMDQ1)