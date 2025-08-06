---
title: YOLOv5
menu:
  default:
    identifier: ko-guides-integrations-yolov5
    parent: integrations
weight: 470
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/yolo/Train_and_Debug_YOLOv5_Models_with_Weights_%26_Biases_.ipynb" >}}

[Ultralytics 의 YOLOv5](https://ultralytics.com/yolo) ("You Only Look Once") 모델 패밀리는 복잡한 과정을 거치지 않고 신경망을 이용해 실시간 오브젝트 검출을 할 수 있게 해줍니다.

[W&B](https://wandb.com)는 YOLOv5에 바로 연동되어 실험 메트릭 추적, 모델 및 데이터셋 버전 관리, 풍부한 모델 예측 시각화 등 다양한 기능을 제공합니다. **YOLO 실험을 시작하기 전에 한 번만 `pip install` 하면 바로 사용할 수 있습니다.**

{{% alert %}}
모든 W&B 로그 기능은 [PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)와 같은 데이터 병렬 다중 GPU 트레이닝에서 호환됩니다.
{{% /alert %}}

## 핵심 실험 추적하기
`wandb`만 설치하면, W&B의 기본 [로그 기능]({{< relref path="/guides/models/track/log/" lang="ko" >}})이 활성화됩니다. 시스템 메트릭, 모델 메트릭, 그리고 대화형 [Dashboards]({{< relref path="/guides/models/track/workspaces.md" lang="ko" >}})에 기록되는 미디어까지 모두 지원합니다.

```python
pip install wandb
git clone https://github.com/ultralytics/yolov5.git
python yolov5/train.py  # 작은 데이터셋으로 작은 네트워크 트레이닝
```

wandb가 터미널에 출력하는 링크만 따라가면 됩니다.

{{< img src="/images/integrations/yolov5_experiment_tracking.png" alt="이런 차트들을 모두 확인하세요." >}}

## 인테그레이션 커스터마이즈하기

YOLO에 몇 가지 간단한 커맨드라인 인수를 추가하면 W&B의 더 다양한 기능을 사용할 수 있습니다.

* `--save_period`에 숫자를 전달하면, W&B가 매 `save_period` 에포크가 끝날 때마다 [model version]({{< relref path="/guides/core/registry/" lang="ko" >}})을 저장합니다. 모델 버전에는 모델 가중치가 포함되고, 검증 세트에서 가장 성능이 우수한 모델엔 태그가 달립니다.
* `--upload_dataset` 플래그를 켜면 데이터셋도 업로드되어 데이터 버전 관리가 가능합니다.
* `--bbox_interval`에 숫자를 입력하면 [데이터 시각화]({{< relref path="../" lang="ko" >}})가 활성화됩니다. 매 `bbox_interval` 에포크마다, 모델이 검증 세트에서 예측한 결과가 W&B로 업로드됩니다.

{{< tabpane text=true >}}
{{% tab header="Model Versioning Only" value="modelversioning" %}}

```python
python yolov5/train.py --epochs 20 --save_period 1
```

{{% /tab %}}
{{% tab header="Model Versioning and Data Visualization" value="bothversioning" %}}

```python
python yolov5/train.py --epochs 20 --save_period 1 \
  --upload_dataset --bbox_interval 1
```

{{% /tab %}}
{{< /tabpane >}}

{{% alert %}}
모든 W&B 계정에는 데이터셋과 모델을 위한 무료 100 GB 저장소가 제공됩니다.
{{% /alert %}}

실제 화면 예시는 다음과 같습니다.

{{< img src="/images/integrations/yolov5_model_versioning.png" alt="Model versioning" >}}

{{< img src="/images/integrations/yolov5_data_visualization.png" alt="Data visualization" >}}

{{% alert %}}
데이터와 모델 버전 관리를 활용하면 멈추거나 중단된 실험도 어떤 기기에서든 별도 설정 없이 바로 이어서 계속할 수 있습니다. 자세한 내용은 [Colab ](https://wandb.me/yolo-colab)에서 확인하세요.
{{% /alert %}}