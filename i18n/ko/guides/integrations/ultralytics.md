---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# Ultralytics

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/01_train_val.ipynb"></CTAButtons>

[Ultralytics](https://github.com/ultralytics/ultralytics)는 이미지 분류, 오브젝트 검출, 이미지 세스멘테이션, 포즈 추정과 같은 작업을 위한 최첨단의 컴퓨터 비전 모델들이 모여 있는 곳입니다. 실시간 오브젝트 검출 모델 시리즈인 [YOLOv8](https://docs.ultralytics.com/models/yolov8/)의 최신 버전 뿐만 아니라, [SAM (Segment Anything Model)](https://docs.ultralytics.com/models/sam/#introduction-to-sam-the-segment-anything-model), [RT-DETR](https://docs.ultralytics.com/models/rtdetr/), [YOLO-NAS](https://docs.ultralytics.com/models/yolo-nas/) 등 다른 강력한 컴퓨터 비전 모델들도 호스팅합니다. Ultralytics는 이러한 모델들의 구현을 제공할 뿐만 아니라, 사용하기 쉬운 API를 사용하여 이 모델들을 트레이닝, 파인튜닝 및 적용하기 위한 엔드투엔드 워크플로우를 제공합니다.

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

**안내:** 통합은 현재 `ultralyticsv8.0.238` 및 이하 버전에서 테스트되었습니다. 문제가 발생하면 https://github.com/wandb/wandb/issues 에 `yolov8` 태그로 문의해 주세요.

## 실험 추적 및 검증 결과 시각화

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/01_train_val.ipynb"></CTAButtons>

이 섹션은 [Ultralytics](https://docs.ultralytics.com/modes/predict/) 모델을 사용하여 트레이닝, 파인튜닝, 검증을 수행하고 실험 추적, 모델 체크포인팅, 모델 성능의 시각화를 [W&B](https://wandb.ai/site)를 사용하여 수행하는 전형적인 워크플로우를 보여줍니다.

Google Colab에서 코드를 시도해 볼 수 있습니다: [Open In Colab](http://wandb.me/ultralytics-train)

이 통합에 대한 리포트도 확인해 보세요: [Supercharging Ultralytics with W&B](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)

Ultralytics와 W&B 통합을 사용하기 위해서는 `wandb.integration.ultralytics.add_wandb_callback` 함수를 import해야 합니다.

```python
import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics import YOLO
```

다음으로, 선택한 `YOLO` 모델을 초기화하고, 모델로 추론을 수행하기 전에 이 모델에 `add_wandb_callback` 함수를 호출합니다. 이렇게 하면 트레이닝, 파인튜닝, 검증 또는 추론을 수행할 때 자동으로 실험 로그와 W&B의 [컴퓨터 비전 작업을 위한 대화형 오버레이](../track/log/media#image-overlays-in-tables)를 사용하여 실제값과 예측값이 오버레이된 이미지와 추가적인 인사이트를 [`wandb.Table`](../tables/intro.md)에 자동으로 로깅합니다.

```python
# YOLO 모델 초기화
model = YOLO("yolov8n.pt")

# Ultralytics를 위한 W&B 콜백 추가
add_wandb_callback(model, enable_model_checkpointing=True)

# 모델을 트레이닝/파인튜닝합니다
# 각 에포크의 끝에, 검증 배치의 예측값이 컴퓨터 비전 작업을 위한
# 통찰력 있고 대화형 오버레이를 갖춘 W&B 테이블에 로깅됩니다
model.train(project="ultralytics", data="coco128.yaml", epochs=5, imgsz=640)

# W&B run을 종료합니다
wandb.finish()
```

W&B를 사용하여 추적된 Ultralytics 트레이닝 또는 파인튜닝 워크플로우의 실험이 어떻게 보이는지 여기에서 확인할 수 있습니다:

<blockquote class="imgur-embed-pub" lang="en" data-id="a/TB76U9O"  ><a href="//imgur.com/a/TB76U9O">YOLO 파인튜닝 실험</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

에포크별 검증 결과가 [W&B 테이블](../tables/intro.md)을 사용하여 어떻게 시각화되는지 여기에서 확인할 수 있습니다:

<blockquote class="imgur-embed-pub" lang="en" data-id="a/kU5h7W4"  ><a href="//imgur.com/a/kU5h7W4">WandB 검증 시각화 테이블</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

## 예측 결과 시각화

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/00_inference.ipynb"></CTAButtons>

이 섹션은 [Ultralytics](https://docs.ultralytics.com/modes/predict/) 모델을 사용하여 추론을 수행하고 [W&B](https://wandb.ai/site)를 사용하여 결과를 시각화하는 전형적인 워크플로우를 보여줍니다.

Google Colab에서 코드를 시도해 볼 수 있습니다: [Open in Colab](http://wandb.me/ultralytics-inference).

이 통합에 대한 리포트도 확인해 보세요: [Supercharging Ultralytics with W&B](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)

Ultralytics와 W&B 통합을 사용하기 위해서는 `wandb.integration.ultralytics.add_wandb_callback` 함수를 import해야 합니다.

```python
import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics.engine.model import YOLO
```

이제 통합을 테스트할 몇 가지 이미지를 다운로드해 보겠습니다. 자신의 이미지, 비디오 또는 카메라 소스를 사용할 수 있습니다. 추론 소스에 대한 자세한 정보는 [공식 문서](https://docs.ultralytics.com/modes/predict/)를 참조하세요.

```python
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img1.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img2.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img4.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img5.png
```

다음으로, `wandb.init`를 사용하여 W&B [run](../runs/intro.md)을 초기화합니다.

```python
# W&B run 초기화
wandb.init(project="ultralytics", job_type="inference")
```

다음으로, 선택한 `YOLO` 모델을 초기화하고, 모델로 추론을 수행하기 전에 이 모델에 `add_wandb_callback` 함수를 호출합니다. 이렇게 하면 추론을 수행할 때 [컴퓨터 비전 작업을 위한 대화형 오버레이](../track/log/media#image-overlays-in-tables)가 오버레이된 이미지와 추가적인 인사이트가 [`wandb.Table`](../tables/intro.md)에 자동으로 로깅됩니다.

```python
# YOLO 모델 초기화
model = YOLO("yolov8n.pt")

# Ultralytics를 위한 W&B 콜백 추가
add_wandb_callback(model, enable_model_checkpointing=True)

# 바운딩 박스, 세그멘테이션 마스크를 위한 대화형 오버레이가 포함된
# W&B 테이블로 자동 로깅되는 예측을 수행합니다
model(["./assets/img1.jpeg", "./assets/img3.png", "./assets/img4.jpeg", "./assets/img5.jpeg"])

# W&B run을 종료합니다
wandb.finish()
```

참고: 트레이닝이나 파인튜닝 워크플로우의 경우 `wandb.init()`을 사용하여 명시적으로 run을 초기화할 필요가 없습니다. 하지만, 코드가 추론만 포함하는 경우에는 run을 명시적으로 생성해야 합니다.

대화형 bbox 오버레이가 어떻게 보이는지 여기에서 확인할 수 있습니다:

<blockquote class="imgur-embed-pub" lang="en" data-id="a/UTSiufs"  ><a href="//imgur.com/a/UTSiufs">WandB 이미지 오버레이</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

W&B 이미지 오버레이에 대한 자세한 정보는 [여기](../track/log/media.md#image-overlays)에서 확인할 수 있습니다.

## 추가 자료

* [Supercharging Ultralytics with Weights & Biases](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)
* [YOLOv8를 사용한 오브젝트 검출: 엔드투엔드 워크플로우](https://wandb.ai/reviewco/object-detection-bdd/reports/Object-Detection-using-YOLOv8-An-End-to-End-Workflow--Vmlldzo1NTAyMDQ1)