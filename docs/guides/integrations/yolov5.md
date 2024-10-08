---
title: YOLOv5
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

[Ultralytics' YOLOv5](https://ultralytics.com/yolov5) ("You Only Look Once") 모델 군은 실시간 오브젝트 검출을 위한 컨벌루션 신경망을 통해 고통 없이 가능합니다.

[Weights & Biases](http://wandb.com)는 YOLOv5에 직접 인테그레이션되어, 실험 메트릭 추적, 모델 및 데이터셋 버전 관리, 풍부한 모델 예측값 시각화 등을 제공합니다. **YOLO 실험을 실행하기 전에 `pip install` 한 번으로 쉽게 사용할 수 있습니다!**

:::안내
YOLOv5 인테그레이션의 모델 및 데이터 로그 기능에 대한 빠른 개요를 보려면, 아래 링크된 [이 Colab](https://wandb.me/yolo-colab)과 동영상 튜토리얼을 확인하세요.
:::

:::안내
모든 W&B 로깅 기능은 데이터 병렬 멀티-GPU 트레이닝과 호환됩니다. 예를 들어 [PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)와 함께 사용할 수 있습니다.
:::

## 핵심 실험 추적

`wandb`를 설치하기만 하면, 시스템 메트릭, 모델 메트릭 및 인터랙티브 [대시보드](../track/app.md)에 로그되는 미디어와 같은 W&B [로깅 기능들](../track/log/intro.md)이 활성화됩니다.

```python
pip install wandb
git clone https://github.com/ultralytics/yolov5.git
python yolov5/train.py  # 작은 네트워크와 작은 데이터셋에서 트레이닝
```

표준 출력에 인쇄된 링크를 따라가면 됩니다.

![모든 차트와 더 많은 것들!](/images/integrations/yolov5_experiment_tracking.png)

## 모델 버전 관리 및 데이터 시각화

하지만 이것만이 전부가 아닙니다! 몇 개의 간단한 커맨드라인 인수를 YOLO에 전달하면 더 많은 W&B 기능을 활용할 수 있습니다.

* `--save_period`에 숫자를 전달하면 [모델 버전 관리](../model_registry/intro.md)가 활성화됩니다. 모든 `save_period` 에포크 끝에 모델 가중치가 W&B에 저장됩니다. 검증 세트에서 최고의 성능을 보인 모델은 자동으로 태그가 설정됩니다.
* `--upload_dataset` 플래그를 켜면 데이터셋도 데이터 버전 관리에 업로드됩니다.
* `--bbox_interval`에 숫자를 전달하면 [데이터 시각화](../intro.md)가 활성화됩니다. 모든 `bbox_interval` 에포크 끝에 모델의 검증 세트에 대한 출력값이 W&B에 업로드됩니다.

<Tabs
  defaultValue="modelversioning"
  values={[
    {label: '모델 버전 관리만', value: 'modelversioning'},
    {label: '모델 버전 관리와 데이터 시각화', value: 'bothversioning'},
  ]}>
  <TabItem value="modelversioning">

```python
python yolov5/train.py --epochs 20 --save_period 1
```

  </TabItem>
  <TabItem value="bothversioning">

```python
python yolov5/train.py --epochs 20 --save_period 1 \
  --upload_dataset --bbox_interval 1
```

  </TabItem>
</Tabs>

:::안내
모든 W&B 계정에는 데이터셋과 모델을 위한 100 GB의 무료 스토리지가 제공됩니다.
:::

이것이 어떻게 보이는지 한번 보세요.

![모델 버전 관리: 최신 버전과 최고의 버전의 모델이 식별됩니다.](/images/integrations/yolov5_model_versioning.png)

![데이터 시각화: 입력 이미지를 모델의 출력값 및 예제별 메트릭과 비교합니다.](/images/integrations/yolov5_data_visualization.png)

:::안내
데이터와 모델 버전 관리로 인해, 중단되거나 충돌된 실험을 어떤 장치에서도 재개할 수 있으며, 추가 설정이 필요 없습니다! 자세한 내용은 [Colab](https://wandb.me/yolo-colab)을 확인하세요.
:::