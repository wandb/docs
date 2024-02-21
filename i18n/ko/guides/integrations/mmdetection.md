---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# MMDetection

[**여기에서 Colab 노트북으로 시도해 보세요 →**](https://github.com/wandb/examples/blob/master/colabs/mmdetection/Train\_an\_Object\_Detection%2BSemantic\_Segmentation\_Model\_with\_MMDetection\_and\_W%26B.ipynb)

[MMDetection](https://github.com/open-mmlab/mmdetection/)은 PyTorch 기반의 오픈 소스 오브젝트 디텍션 도구 모음으로, [OpenMMLab](https://openmmlab.com/)의 일부입니다. 이는 조합 가능하고 모듈식 API 디자인을 제공하여 사용자가 맞춤형 오브젝트 디텍션 및 세그멘테이션 파이프라인을 쉽게 구축할 수 있습니다.

[Weights and Biases](https://wandb.ai/site)는 전용 `MMDetWandbHook`을 통해 MMDetection에 직접 통합되어 다음을 수행할 수 있습니다:

✅ 학습 및 평가 메트릭 로그.

✅ 버전 관리된 모델 체크포인트 로그.

✅ 실제값 바운딩 박스가 있는 버전 관리된 검증 데이터세트 로그.

✅ 모델 예측 로그 및 시각화.

## :fire: 시작하기

### wandb에 가입하고 로그인하기

a) [**무료 계정에 가입하기**](https://wandb.ai/site)

b) `wandb` 라이브러리를 Pip으로 설치하기

c) 학습 스크립트에서 로그인하려면 www.wandb.ai에 로그인한 상태여야 하며, 그 후 **API 키를** [**인증 페이지에서 찾을 수 있습니다.**](https://wandb.ai/authorize)**.**

Weights and Biases를 처음 사용하는 경우 [퀵스타트](../../quickstart.md)를 확인해 보세요.

<Tabs
  defaultValue="cli"
  values={[
    {label: '명령 줄', value: 'cli'},
    {label: '노트북', value: 'notebook'},
  ]}>
  <TabItem value="cli">


```bash
pip install wandb
wandb login
```


  </TabItem>
  <TabItem value="notebook">


```notebook
!pip install wandb
wandb.login()
```


  </TabItem>
</Tabs>

### `MMDetWandbHook` 사용하기

`MMDetWandbHook`을 MMDetection의 `log_config` 메서드 구성 시스템에 추가하여 Weights and Biases를 사용하기 시작할 수 있습니다.

:::안내
`MMDetWandbHook`은 [MMDetection v2.25.0](https://twitter.com/OpenMMLab/status/1532193548283432960?s=20\&t=dzBiKn9dlNdrvK8e\_q0zfQ) 이상에서 지원됩니다.
:::

```python
import wandb

...

config_file = "mmdetection/configs/path/to/config.py"
cfg = Config.fromfile(config_file)

cfg.log_config.hooks = [
    dict(type="TextLoggerHook"),
    dict(
        type="MMDetWandbHook",
        init_kwargs={"project": "mmdetection"},
        interval=10,
        log_checkpoint=True,
        log_checkpoint_metadata=True,
        num_eval_images=100,
        bbox_score_thr=0.3,
    ),
]
```

| 이름                      | 설명                                                                                                                                                             |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `init_kwargs`             | (`dict`) wandb.init에 전달되어 W&B 실행을 초기화하는 데 사용되는 사전입니다.                                                                                                          |
| `interval`                | (`int`) 로깅 간격(모든 k 반복마다). 기본값은 `50`입니다.                                                                                                        |
| `log_checkpoint`          | (`bool`) 모든 체크포인트 간격에서 체크포인트를 W&B 아티팩트로 저장합니다. 이는 모델 버전 관리에 사용됩니다. 기본값은 `False`입니다.     |
| `log_checkpoint_metadata` | (`bool`) 검증 데이터에서 계산된 평가 메트릭을 현재 에포크와 함께 해당 체크포인트의 메타데이터로 로그합니다. 기본값은 `True`입니다. |
| `num_eval_images`         | (`int`) 로그된 검증 이미지의 수입니다. 0이면 평가가 로그되지 않습니다. 기본값은 `100`입니다.                                                       |
| `bbox_score_thr`          | (`float`) 바운딩 박스 점수의 임계값입니다. 기본값은 `0.3`입니다.                                                                                                         |

### :chart\_with\_upwards\_trend: 메트릭 로그

`MMDetWandbHook`의 `init_kwargs` 인수를 사용하여 학습 및 평가 메트릭 추적을 시작합니다. 이 인수는 키-값 쌍의 사전을 받아들이며 이는 `wandb.init`에 전달되어 실행이 로그되는 프로젝트 및 실행의 다른 기능을 제어합니다.

```
init_kwargs={
    'project': 'mmdetection',
    'entity': 'my_team_name',
    'config': {'lr': 1e-4, 'batch_size':32},
    'tags': ['resnet50', 'sgd'] 
}
```

wandb.init의 모든 인수는 [여기](https://docs.wandb.ai/ref/python/init)에서 확인하세요.

![](@site/static/images/integrations/log_metrics.gif)

### :checkered\_flag: 체크포인팅

`MMDetWandbHook`의 `log_checkpoint=True` 인수를 사용하여 이러한 체크포인트를 [W&B 아티팩트](../artifacts/intro.md)로 안정적으로 저장할 수 있습니다. 이 기능은 MMCV의 [`CheckpointHook`](https://mmcv.readthedocs.io/en/latest/api.html?highlight=CheckpointHook#mmcv.runner.CheckpointHook)에 의존하며, 이는 주기적으로 모델 체크포인트를 저장합니다. 주기는 `checkpoint_config.interval`에 의해 결정됩니다.

:::안내
모든 W&B 계정에는 데이터세트와 모델을 위한 100GB의 무료 저장 공간이 제공됩니다.
:::

![체크포인트는 왼쪽 창에 다른 버전으로 표시됩니다. 모델은 파일 탭에서 다운로드하거나 API를 사용하여 프로그래매틱하게 다운로드할 수 있습니다.](/images/integrations/mmdetection_checkpointing.png)

### 메타데이터와 함께 체크포인트

`log_checkpoint_metadata`가 `True`라면, 모든 체크포인트 버전에는 메타데이터가 연결됩니다. 이 기능은 `CheckpointHook`과 `EvalHook` 또는 `DistEvalHook`에 의존합니다. 메타데이터는 체크포인트 간격이 평가 간격으로 나누어질 때만 로그됩니다.

![로그된 메타데이터는 메타데이터 탭에 표시됩니다.](@site/static/images/integrations/mmdetection_checkpoint_metadata.png)

### 데이터세트 및 모델 예측 시각화

데이터세트, 특히 모델 예측을 대화형으로 시각화할 수 있는 능력은 더 나은 모델을 구축하고 디버그하는 데 도움이 됩니다. `MMDetWandbHook`을 사용하면 검증 데이터를 W&B 테이블로 로그하고 모델 예측에 대한 버전 관리된 W&B 테이블을 생성할 수 있습니다.

`num_eval_images` 인수는 W&B 테이블로 로그되는 검증 샘플의 수를 제어합니다. 주의할 몇 가지 사항은 다음과 같습니다:

* `num_eval_images=0`이면 검증 데이터와 모델 예측이 로그되지 않습니다.
* [`mmdet.core.train_detector`](https://mmdetection.readthedocs.io/en/latest/\_modules/mmdet/apis/train.html?highlight=train\_detector) API에 대해 `validate=False`인 경우, 검증 데이터와 모델 예측이 로그되지 않습니다.
* `num_eval_images`가 검증 샘플의 총 수보다 큰 경우, 전체 검증 데이터세트가 로그됩니다.



:::안내
`val_data`는 한 번만 업로드됩니다. `run_<id>_pred` 테이블과 후속 실행은 업로드된 데이터를 참조하여 메모리를 절약합니다. `val_data`가 변경될 때만 새로운 버전이 생성됩니다.
:::

## 다음 단계

사용자 정의 데이터세트에서 인스턴스 세그멘테이션 모델(Mask R-CNN)을 학습하려면 [Weights & Biases와 MMDetection 사용 방법](https://wandb.ai/ayush-thakur/mmdetection/reports/How-to-Use-Weights-Biases-with-MMDetection--VmlldzoyMTM0MDE2) W&B 리포트를 확인해 보세요. [Fully Connected](https://wandb.ai/fully-connected)에서 확인할 수 있습니다.

이 Weights & Biases 통합에 대한 질문이나 문제가 있으신가요? [MMDetection github 저장소](https://github.com/open-mmlab/mmdetection)에 문제를 제기하시면 답변을 드리겠습니다 :)