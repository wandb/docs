---
title: Ultralytics
menu:
  default:
    identifier: ko-guides-integrations-ultralytics
    parent: integrations
weight: 480
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/01_train_val.ipynb" >}}

[Ultralytics](https://github.com/ultralytics/ultralytics)는 이미지 분류, 오브젝트 검출, 이미지 세분화 및 포즈 추정과 같은 작업을 위한 최첨단 컴퓨터 비전 모델의 본거지입니다. 실시간 오브젝트 검출 모델인 YOLO 시리즈의 최신 반복인 [YOLOv8](https://docs.ultralytics.com/models/yolov8/)뿐만 아니라 [SAM (Segment Anything Model)](https://docs.ultralytics.com/models/sam/#introduction-to-sam-the-segment-anything-model), [RT-DETR](https://docs.ultralytics.com/models/rtdetr/), [YOLO-NAS](https://docs.ultralytics.com/models/yolo-nas/) 등과 같은 다른 강력한 컴퓨터 비전 모델도 호스팅합니다. Ultralytics는 이러한 모델의 구현을 제공하는 것 외에도 사용하기 쉬운 API를 사용하여 이러한 모델을 트레이닝, 파인튜닝 및 적용할 수 있는 즉시 사용 가능한 워크플로우를 제공합니다.

## 시작하기

1. `ultralytics` 와 `wandb`를 설치합니다.

    {{< tabpane text=true >}}
    {{% tab header="커맨드 라인" value="script" %}}

    ```shell
    pip install --upgrade ultralytics==8.0.238 wandb

    # or
    # conda install ultralytics
    ```

    {{% /tab %}}
    {{% tab header="노트북" value="notebook" %}}

    ```bash
    !pip install --upgrade ultralytics==8.0.238 wandb
    ```

    {{% /tab %}}
    {{< /tabpane >}}

    개발팀은 `ultralyticsv8.0.238` 이하 버전과의 통합을 테스트했습니다. 통합에 대한 문제가 있으면 `yolov8` 태그를 사용하여 [GitHub issue](https://github.com/wandb/wandb/issues/new?template=sdk-bug.yml)를 생성하세요.

## Experiments 추적 및 검증 결과 시각화

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/01_train_val.ipynb" >}}

이 섹션에서는 트레이닝, 파인튜닝 및 검증을 위해 [Ultralytics](https://docs.ultralytics.com/modes/predict/) 모델을 사용하고, [W&B](https://wandb.ai/site)를 사용하여 experiment 추적, 모델-체크포인트, 모델 성능 시각화를 수행하는 일반적인 워크플로우를 보여줍니다.

다음 리포트에서 통합에 대해 확인할 수도 있습니다: [W&B로 Ultralytics 강화](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)

Ultralytics와 W&B 통합을 사용하려면 `wandb.integration.ultralytics.add_wandb_callback` 함수를 가져옵니다.

```python
import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics import YOLO
```

선택한 `YOLO` 모델을 초기화하고 모델로 추론을 수행하기 전에 `add_wandb_callback` 함수를 호출합니다. 이렇게 하면 트레이닝, 파인튜닝, 검증 또는 추론을 수행할 때 experiment 로그와 이미지가 자동으로 저장되고, [컴퓨터 비전 작업을 위한 대화형 오버레이]({{< relref path="/guides/models/track/log/media#image-overlays-in-tables" lang="ko" >}})를 사용하여 각각의 예측 결과와 함께 W&B의 [`wandb.Table`]({{< relref path="/guides/models/tables/" lang="ko" >}})에 추가 인사이트와 함께 오버레이됩니다.

```python
# YOLO 모델 초기화
model = YOLO("yolov8n.pt")

# Ultralytics에 W&B 콜백 추가
add_wandb_callback(model, enable_model_checkpointing=True)

# 모델 트레이닝/파인튜닝
# 각 에포크가 끝나면 검증 배치에 대한 예측이 기록됩니다.
# 컴퓨터 비전 작업을 위한 통찰력 있고 상호 작용적인 오버레이가 있는 W&B 테이블에
model.train(project="ultralytics", data="coco128.yaml", epochs=5, imgsz=640)

# W&B run 종료
wandb.finish()
```

다음은 Ultralytics 트레이닝 또는 파인튜닝 워크플로우에 대해 W&B를 사용하여 추적된 Experiments의 모습입니다.

<blockquote class="imgur-embed-pub" lang="en" data-id="a/TB76U9O"  ><a href="//imgur.com/a/TB76U9O">YOLO Fine-tuning Experiments</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

다음은 [W&B Table]({{< relref path="/guides/models/tables/" lang="ko" >}})을 사용하여 에포크별 검증 결과를 시각화하는 방법입니다.

<blockquote class="imgur-embed-pub" lang="en" data-id="a/kU5h7W4"  ><a href="//imgur.com/a/kU5h7W4">WandB Validation Visualization Table</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

## 예측 결과 시각화

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/00_inference.ipynb" >}}

이 섹션에서는 추론을 위해 [Ultralytics](https://docs.ultralytics.com/modes/predict/) 모델을 사용하고 [W&B](https://wandb.ai/site)를 사용하여 결과를 시각화하는 일반적인 워크플로우를 보여줍니다.

Google Colab에서 코드를 사용해 볼 수 있습니다: [Colab에서 열기](http://wandb.me/ultralytics-inference).

다음 리포트에서 통합에 대해 확인할 수도 있습니다: [W&B로 Ultralytics 강화](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)

Ultralytics와 W&B 통합을 사용하려면 `wandb.integration.ultralytics.add_wandb_callback` 함수를 가져와야 합니다.

```python
import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics.engine.model import YOLO
```

통합을 테스트할 이미지를 몇 개 다운로드합니다. 스틸 이미지, 비디오 또는 카메라 소스를 사용할 수 있습니다. 추론 소스에 대한 자세한 내용은 [Ultralytics 문서](https://docs.ultralytics.com/modes/predict/)를 확인하세요.

```bash
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img1.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img2.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img4.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img5.png
```

다음으로 `wandb.init`을 사용하여 W&B [run]({{< relref path="/guides/models/track/runs/" lang="ko" >}})을 초기화합니다.

```python
# W&B run 초기화
wandb.init(project="ultralytics", job_type="inference")
```

다음으로 원하는 `YOLO` 모델을 초기화하고 모델로 추론을 수행하기 전에 `add_wandb_callback` 함수를 호출합니다. 이렇게 하면 추론을 수행할 때 [컴퓨터 비전 작업을 위한 대화형 오버레이]({{< relref path="/guides/models/track/log/media#image-overlays-in-tables" lang="ko" >}})와 함께 [`wandb.Table`]({{< relref path="/guides/models/tables/" lang="ko" >}})에 추가 인사이트와 함께 이미지가 자동으로 기록됩니다.

```python
# YOLO 모델 초기화
model = YOLO("yolov8n.pt")

# Ultralytics에 W&B 콜백 추가
add_wandb_callback(model, enable_model_checkpointing=True)

# W&B 테이블에 자동으로 기록되는 예측 수행
# 경계 상자, 세분화 마스크에 대한 대화형 오버레이 포함
model(
    [
        "./assets/img1.jpeg",
        "./assets/img3.png",
        "./assets/img4.jpeg",
        "./assets/img5.jpeg",
    ]
)

# W&B run 종료
wandb.finish()
```

트레이닝 또는 파인튜닝 워크플로우의 경우 `wandb.init()`을 사용하여 명시적으로 run을 초기화할 필요가 없습니다. 그러나 코드에 예측만 포함된 경우 run을 명시적으로 생성해야 합니다.

다음은 대화형 bbox 오버레이의 모습입니다.

<blockquote class="imgur-embed-pub" lang="en" data-id="a/UTSiufs"  ><a href="//imgur.com/a/UTSiufs">WandB Image Overlay</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

W&B 이미지 오버레이에 대한 자세한 내용은 [여기]({{< relref path="/guides/models/track/log/media.md#image-overlays" lang="ko" >}})에서 확인할 수 있습니다.

## 추가 자료

* [Weights & Biases로 Ultralytics 강화](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)
* [YOLOv8을 사용한 오브젝트 검출: 엔드투엔드 워크플로우](https://wandb.ai/reviewco/object-detection-bdd/reports/Object-Detection-using-YOLOv8-An-End-to-End-Workflow--Vmlldzo1NTAyMDQ1)
