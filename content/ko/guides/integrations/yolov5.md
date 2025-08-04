---
title: YOLOv5
menu:
  default:
    identifier: ko-guides-integrations-yolov5
    parent: integrations
weight: 470
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/yolo/Train_and_Debug_YOLOv5_Models_with_Weights_%26_Biases_.ipynb" >}}

[Ultralytics' YOLOv5](https://ultralytics.com/yolo) ("You Only Look Once") 모델 제품군은 고통스러운 과정 없이 컨볼루션 신경망을 통해 실시간 오브젝트 검출을 가능하게 합니다.

[Weights & Biases](http://wandb.com)는 YOLOv5에 직접 통합되어 실험 메트릭 추적, 모델 및 데이터셋 버전 관리, 풍부한 모델 예측 시각화 등을 제공합니다. **YOLO 실험을 실행하기 전에 간단히 `pip install` 한 번만 실행하면 됩니다.**

{{% alert %}}
모든 W&B 로깅 기능은 [PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)와 같은 데이터 병렬 멀티 GPU 트레이닝과 호환됩니다.
{{% /alert %}}

## 핵심 Experiments 추적
`wandb`를 설치하는 것만으로도 W&B 기본 제공 [로깅 기능]({{< relref path="/guides/models/track/log/" lang="ko" >}})을 활성화할 수 있습니다. 시스템 메트릭, 모델 메트릭 및 대화형 [대시보드]({{< relref path="/guides/models/track/workspaces.md" lang="ko" >}})에 기록된 미디어를 확인할 수 있습니다.

```python
pip install wandb
git clone https://github.com/ultralytics/yolov5.git
python yolov5/train.py  # 작은 데이터셋에서 작은 네트워크를 트레이닝합니다.
```

wandb에서 표준 출력으로 출력된 링크를 따라가기만 하면 됩니다.

{{< img src="/images/integrations/yolov5_experiment_tracking.png" alt="이러한 모든 차트 등을 사용할 수 있습니다." >}}

## 통합 사용자 정의

YOLO에 몇 가지 간단한 커맨드라인 인수를 전달하여 더 많은 W&B 기능을 활용할 수 있습니다.

* `--save_period`에 숫자를 전달하면 W&B는 모든 `save_period` 에포크가 끝날 때마다 [모델 버전]({{< relref path="/guides/core/registry/" lang="ko" >}})을 저장합니다. 모델 버전에는 모델 가중치가 포함되어 있으며 검증 세트에서 가장 성능이 좋은 모델에 태그를 지정합니다.
* `--upload_dataset` 플래그를 켜면 데이터 버전 관리를 위해 데이터셋도 업로드됩니다.
* `--bbox_interval`에 숫자를 전달하면 [데이터 시각화]({{< relref path="../" lang="ko" >}})이 켜집니다. 모든 `bbox_interval` 에포크가 끝나면 검증 세트에서 모델의 출력이 W&B에 업로드됩니다.

{{< tabpane text=true >}}
{{% tab header="모델 버전 관리만 해당" value="modelversioning" %}}

```python
python yolov5/train.py --epochs 20 --save_period 1
```

{{% /tab %}}
{{% tab header="모델 버전 관리 및 Data Visualization" value="bothversioning" %}}

```python
python yolov5/train.py --epochs 20 --save_period 1 \
  --upload_dataset --bbox_interval 1
```

{{% /tab %}}
{{< /tabpane >}}

{{% alert %}}
모든 W&B 계정에는 데이터셋 및 모델을 위한 100GB의 무료 스토리지가 제공됩니다.
{{% /alert %}}

다음은 그 모습입니다.

{{< img src="/images/integrations/yolov5_model_versioning.png" alt="모델 버전 관리: 최신 버전과 가장 우수한 버전의 모델이 식별됩니다." >}}

{{< img src="/images/integrations/yolov5_data_visualization.png" alt="Data Visualization: 입력 이미지를 모델의 출력 및 예제별 메트릭과 비교합니다." >}}

{{% alert %}}
데이터 및 모델 버전 관리를 사용하면 설정 없이 모든 장치에서 일시 중지되거나 충돌된 Experiments를 재개할 수 있습니다. 자세한 내용은 [Colab](https://wandb.me/yolo-colab)을 확인하세요.
{{% /alert %}}
