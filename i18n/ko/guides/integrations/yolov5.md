---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# YOLOv5

[Ultralytics' YOLOv5](https://ultralytics.com/yolov5) ("한 번만 보세요") 모델 패밀리는 모든 고통스러운 부분 없이 실시간 오브젝트 검출을 컨볼루셔널 신경망으로 가능하게 합니다.

[Weights & Biases](http://wandb.com)는 YOLOv5에 직접 통합되어 실험 메트릭 추적, 모델 및 데이터셋 버전 관리, 풍부한 모델 예측 시각화 등을 제공합니다. **YOLO 실험을 실행하기 전에 단 하나의 `pip install`을 실행하는 것만큼 쉽습니다!**

:::info
모델 및 데이터 로깅 기능의 빠른 개요를 보려면 아래 링크된 [이 Colab](https://wandb.me/yolo-colab) 및 동영상 튜토리얼을 확인하세요.
:::

:::info
모든 W&B 로깅 기능은 데이터 병렬 멀티 GPU 트레이닝, 예를 들어 [PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)와 호환됩니다.
:::

## 핵심 실험 추적

`wandb`를 설치하기만 하면 W&B의 내장 [로그 기능](../track/log/intro.md)이 활성화됩니다: 시스템 메트릭, 모델 메트릭 및 미디어가 상호작용 가능한 [대시보드](../track/app.md)에 로그됩니다.

```python
pip install wandb
git clone https://github.com/ultralytics/yolov5.git
python yolov5/train.py  # 작은 데이터셋에서 작은 네트워크를 트레이닝합니다
```

wandb가 표준 출력에 출력하는 링크를 따라가기만 하면 됩니다.

![모든 이 차트들과 더 많은 것들!](/images/integrations/yolov5_experiment_tracking.png)

## 모델 버전 관리 및 데이터 시각화

하지만 그게 전부가 아닙니다! YOLO에 몇 가지 간단한 커맨드라인 인수를 전달함으로써 더 많은 W&B 기능을 활용할 수 있습니다.

* `--save_period`에 숫자를 전달하면 [모델 버전 관리](../model_registry/intro.md)가 활성화됩니다. 모든 `save_period` 에포크마다 모델 가중치가 W&B에 저장됩니다. 검증 세트에서 가장 성능이 좋은 모델이 자동으로 태그됩니다.
* `--upload_dataset` 플래그를 켜면 데이터셋도 데이터 버전 관리를 위해 업로드됩니다.
* `--bbox_interval`에 숫자를 전달하면 [데이터 시각화](../intro.md)가 활성화됩니다. 모든 `bbox_interval` 에포크마다 모델의 검증 세트에 대한 출력이 W&B에 업로드됩니다.

<Tabs
  defaultValue="modelversioning"
  values={[
    {label: '모델 버전 관리만', value: 'modelversioning'},
    {label: '모델 버전 관리 및 데이터 시각화', value: 'bothversioning'},
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

:::info
모든 W&B 계정에는 데이터셋 및 모델을 위한 100GB의 무료 저장 공간이 제공됩니다.
:::

이것이 어떻게 보이는지 여기 있습니다.

![모델 버전 관리: 모델의 최신 및 최고 버전이 식별됩니다.](/images/integrations/yolov5_model_versioning.png)

![데이터 시각화: 입력 이미지와 모델의 출력 및 예제별 메트릭을 비교합니다.](/images/integrations/yolov5_data_visualization.png)

:::info
데이터 및 모델 버전 관리를 통해 어떤 장치에서든 실험을 중단하거나 충돌한 실험을 다시 시작할 수 있습니다. 자세한 내용은 [Colab](https://wandb.me/yolo-colab)을 확인하세요.
:::