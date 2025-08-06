---
title: Ultralytics
menu:
  default:
    identifier: ko-guides-integrations-ultralytics
    parent: integrations
weight: 480
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/01_train_val.ipynb" >}}

[Ultralytics](https://github.com/ultralytics/ultralytics)는 이미지 분류, 오브젝트 검출, 이미지 세그멘테이션, 포즈 추정 등 다양한 컴퓨터 비전 태스크를 위한 최첨단 모델을 제공하는 곳입니다. 최신 실시간 오브젝트 검출 모델 시리즈인 [YOLOv8](https://docs.ultralytics.com/models/yolov8/) 뿐만 아니라, [SAM (Segment Anything Model)](https://docs.ultralytics.com/models/sam/#introduction-to-sam-the-segment-anything-model), [RT-DETR](https://docs.ultralytics.com/models/rtdetr/), [YOLO-NAS](https://docs.ultralytics.com/models/yolo-nas/) 등 다양한 강력한 컴퓨터 비전 모델을 함께 사용할 수 있습니다. 이 모델들의 구현체를 제공할 뿐만 아니라, Ultralytics는 사용하기 쉬운 API로 트레이닝, 파인튜닝 및 예측에 바로 쓸 수 있는 워크플로우도 함께 제공합니다.

## 시작하기

1. `ultralytics`와 `wandb`를 설치하세요.

    {{< tabpane text=true >}}
    {{% tab header="커맨드라인" value="script" %}}

    ```shell
    pip install --upgrade ultralytics==8.0.238 wandb

    # 또는
    # conda install ultralytics
    ```

    {{% /tab %}}
    {{% tab header="노트북" value="notebook" %}}

    ```bash
    !pip install --upgrade ultralytics==8.0.238 wandb
    ```

    {{% /tab %}}
    {{< /tabpane >}}

    개발팀은 `ultralyticsv8.0.238` 이하 버전과의 인테그레이션을 테스트했습니다. 인테그레이션 관련 이슈가 있다면, `yolov8` 태그와 함께 [GitHub issue](https://github.com/wandb/wandb/issues/new?template=sdk-bug.yml)를 남겨주세요.

## 실험 추적 및 검증 결과 시각화

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/01_train_val.ipynb" >}}

이 섹션에서는 [Ultralytics](https://docs.ultralytics.com/modes/predict/) 모델을 사용하여 트레이닝, 파인튜닝, 검증을 진행하며 experiment tracking, 모델 체크포인트 저장, 그리고 모델 성능 시각화를 [W&B](https://wandb.ai/site)를 통해 어떻게 진행할 수 있는지 보여줍니다.

인테그레이션에 대한 자세한 내용은 이 리포트도 참고하세요: [Supercharging Ultralytics with W&B](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)

W&B와 Ultralytics를 연동해 사용하려면 `wandb.integration.ultralytics.add_wandb_callback` 함수를 임포트하면 됩니다.

```python
import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics import YOLO
```

원하는 `YOLO` 모델을 초기화한 뒤, inference를 실행하기 전에 `add_wandb_callback`을 등록하세요. 이렇게 하면 트레이닝, 파인튜닝, 검증 또는 inference를 진행할 때 자동으로 experiment 로그와 이미지가 저장되며, 실제 정답과 예측 결과를 모두 [컴퓨터 비전 태스크용 인터랙티브 오버레이]({{< relref path="/guides/models/track/log/media#image-overlays-in-tables" lang="ko" >}})로 W&B에 올릴 수 있습니다. 부가적인 분석도 [`wandb.Table`]({{< relref path="/guides/models/tables/" lang="ko" >}})을 통해 제공됩니다.

```python
with wandb.init(project="ultralytics", job_type="train") as run:

    # YOLO 모델 초기화
    model = YOLO("yolov8n.pt")

    # Ultralytics에 W&B 콜백 추가
    add_wandb_callback(model, enable_model_checkpointing=True)

    # 모델 트레이닝/파인튜닝
    # 각 에포크 종료 시마다 validation 배치에 대한 예측 결과가
    # W&B Table에 직관적이고 인터랙티브한 오버레이와 함께 기록됩니다.
    model.train(project="ultralytics", data="coco128.yaml", epochs=5, imgsz=640)
```

아래는 Ultralytics의 트레이닝 또는 파인튜닝 워크플로우에서 W&B로 실험을 추적한 예시입니다:

<blockquote class="imgur-embed-pub" lang="en" data-id="a/TB76U9O"  ><a href="//imgur.com/a/TB76U9O">YOLO Fine-tuning Experiments</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

에포크별 검증 결과가 [W&B Table]({{< relref path="/guides/models/tables/" lang="ko" >}})에서 어떻게 시각화되는지 살펴보세요:

<blockquote class="imgur-embed-pub" lang="en" data-id="a/kU5h7W4"  ><a href="//imgur.com/a/kU5h7W4">WandB Validation Visualization Table</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

## 예측 결과 시각화

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/00_inference.ipynb" >}}

이 섹션에서는 [Ultralytics](https://docs.ultralytics.com/modes/predict/) 모델로 inference를 진행하고 그 결과를 [W&B](https://wandb.ai/site)로 시각화하는 워크플로우를 보여줍니다.

Google Colab에서 코드를 직접 실행해 볼 수도 있습니다: [Colab에서 열기](https://wandb.me/ultralytics-inference).

인테그레이션에 대한 자세한 내용은 리포트도 참고해보세요: [Supercharging Ultralytics with W&B](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)

W&B와 Ultralytics를 연동하려면 `wandb.integration.ultralytics.add_wandb_callback` 함수를 임포트해야 합니다.

```python
import wandb
from wandb.integration.ultralytics import add_wandb_callback

from ultralytics.engine.model import YOLO
```

인테그레이션을 테스트할 이미지를 몇 개 다운로드하세요. 정지 이미지, 비디오, 카메라 소스 모두 사용할 수 있습니다. inference 소스에 대한 자세한 내용은 [Ultralytics docs](https://docs.ultralytics.com/modes/predict/)를 참고하세요.

```bash
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img1.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img2.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img4.png
!wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img5.png
```

`wandb.init()`을 사용해 W&B [run]({{< relref path="/guides/models/track/runs/" lang="ko" >}})을 시작합니다. 그다음 원하는 `YOLO` 모델을 초기화한 후, inference 전에 `add_wandb_callback`을 등록하세요. 이렇게 하면 inference를 실행할 때마다 [컴퓨터 비전 태스크용 인터랙티브 오버레이]({{< relref path="/guides/models/track/log/media#image-overlays-in-tables" lang="ko" >}})와 부가 분석을 담은 [`wandb.Table`]({{< relref path="/guides/models/tables/" lang="ko" >}})이 자동으로 기록됩니다.

```python
# W&B Run 초기화
with wandb.init(project="ultralytics", job_type="inference") as run:
    # YOLO 모델 초기화
    model = YOLO("yolov8n.pt")

    # Ultralytics에 W&B 콜백 추가
    add_wandb_callback(model, enable_model_checkpointing=True)

    # 예측 실행 - 예측 결과가 W&B Table에 자동으로 기록됨
    # 바운딩박스, 세그멘테이션 마스크 등을 위한 인터랙티브 오버레이 포함
    model(
        [
            "./assets/img1.jpeg",
            "./assets/img3.png",
            "./assets/img4.jpeg",
            "./assets/img5.jpeg",
        ]
    )
```

트레이닝이나 파인튜닝 워크플로우에서는 `wandb.init()`을 명시적으로 실행할 필요가 없습니다. 하지만 예측만 진행하는 코드라면 반드시 run을 생성해야 합니다.

아래는 인터랙티브 bbox 오버레이 예시입니다:

<blockquote class="imgur-embed-pub" lang="en" data-id="a/UTSiufs"  ><a href="//imgur.com/a/UTSiufs">WandB Image Overlay</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>

자세한 내용은 [W&B 이미지 오버레이 가이드]({{< relref path="/guides/models/track/log/media.md#image-overlays" lang="ko" >}})를 읽어보세요.

## 추가 자료

* [Supercharging Ultralytics with W&B](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)
* [Object Detection using YOLOv8: An End-to-End Workflow](https://wandb.ai/reviewco/object-detection-bdd/reports/Object-Detection-using-YOLOv8-An-End-to-End-Workflow--Vmlldzo1NTAyMDQ1)