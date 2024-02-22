---
displayed_sidebar: default
---

# TensorBoard

## 1줄의 코드로 호스팅되는 TensorBoard

Weights & Biases를 사용하면 TensorBoard 로그를 클라우드에 쉽게 업로드하고, 동료 및 동급생과 결과를 빠르게 공유하며 분석을 한 곳에 중앙 집중화할 수 있습니다.

**이 노트북으로 지금 시작하세요:** [**여기서 Colab 노트북에서 시도해 보세요 →**](https://github.com/wandb/examples/blob/master/colabs/tensorboard/TensorBoard\_and\_Weights\_and\_Biases.ipynb)


![](/images/integrations/tensorboard_oneline_code.webp)

### 코드 한 줄만 추가하세요

```python
import wandb

# `sync_tensorboard=True`로 wandb 실행 시작
wandb.init(project="my-project", sync_tensorboard=True)

# TensorBoard를 사용한 학습 코드
...

# [선택적] wandb 실행을 마무리하여 TensorBoard 로그를 W&B에 업로드합니다 (노트북에서 실행하는 경우)
wandb.finish()
```

[**Weights & Biases에서 호스팅되는 Tensorboard의 예제를 여기서 확인하세요**](https://wandb.ai/rymc/simple-tensorboard-example/runs/oab614zf/tensorboard)

wandb 실행이 완료되면, TensorBoard 이벤트 파일이 Weights & Biases에 업로드됩니다. 이 메트릭은 또한 Weights & Biases 차트에 네이티브하게 기록될 것이며, 사용자의 기기의 CPU 또는 GPU 사용량, git 상태, 사용된 터미널 명령어 등 많은 유용한 정보와 함께 기록됩니다.

:::info
Weights & Biases는 TensorFlow의 모든 버전에서 TensorBoard를 지원합니다. W&B는 PyTorch와 함께 사용 시 1.14 이상의 TensorBoard도 지원합니다.
:::

## 자주 묻는 질문

### TensorBoard에 기록되지 않는 메트릭을 W&B에 어떻게 기록하나요?

TensorBoard에 기록되지 않는 추가적인 사용자 정의 메트릭을 기록해야 하는 경우, 코드에서 `wandb.log`를 호출할 수 있습니다 `wandb.log({"custom": 0.8})`

Tensorboard를 동기화할 때 `wandb.log`에서 step 인수를 설정하는 것은 비활성화됩니다. 다른 스텝 수를 설정하고 싶다면, 스텝 메트릭과 함께 메트릭을 기록할 수 있습니다:

`wandb.log({"custom": 0.8, "global_step": global_step})`

### `wandb`를 사용할 때 Tensorboard를 어떻게 구성하나요?

TensorBoard가 패치되는 방식을 더 많이 제어하고 싶다면, `sync_tensorboard=True`를 `wandb.init`에 전달하는 대신 `wandb.tensorboard.patch`를 호출할 수 있습니다.

```python
import wandb

wandb.tensorboard.patch(root_logdir="<logging_directory>")
wandb.init()

# W&B에 TensorBoard 로그를 업로드하기 위해 wandb 실행을 마무리합니다 (노트북에서 실행하는 경우)
wandb.finish()
```

이 메서드에 `tensorboard_x=False`를 전달하여 바닐라 TensorBoard가 패치되도록 할 수 있으며, PyTorch와 함께 TensorBoard > 1.14를 사용하는 경우 `pytorch=True`를 전달하여 패치되도록 할 수 있습니다. 이 옵션들은 가져온 라이브러리의 버전에 따라 스마트한 기본값을 가집니다.

기본적으로, `tfevents` 파일과 모든 `.pbtxt` 파일도 동기화합니다. 이를 통해 대신 TensorBoard 인스턴스를 시작할 수 있습니다. 실행 페이지에서 [TensorBoard 탭](https://www.wandb.com/articles/hosted-tensorboard)을 볼 수 있습니다. 이 동작은 `wandb.tensorboard.patch`에 `save=False`를 전달하여 비활성화할 수 있습니다.

```python
import wandb

wandb.init()
wandb.tensorboard.patch(save=False, tensorboard_x=True)

# 노트북에서 실행하는 경우, W&B에 TensorBoard 로그를 업로드하기 위해 wandb 실행을 마무리합니다
wandb.finish()
```

:::caution
`tf.summary.create_file_writer`를 호출하거나 `torch.utils.tensorboard.SummaryWriter`를 구성하기 **전에** 반드시 `wandb.init` 또는 `wandb.tensorboard.patch`를 호출해야 합니다.
:::

### 이전 TensorBoard 실행 동기화하기

로컬에 저장된 기존 `tfevents` 파일을 W&B로 가져오고 싶다면, `wandb sync log_dir`을 실행할 수 있습니다. 여기서 `log_dir`은 `tfevents` 파일이 포함된 로컬 디렉터리입니다.

### Google Colab, Jupyter 및 TensorBoard

Jupyter 또는 Colab 노트북에서 코드를 실행하는 경우, 학습이 끝난 후 `wandb.finish()`를 반드시 호출하세요. 이렇게 하면 wandb 실행이 마무리되고 tensorboard 로그가 W&B에 업로드되어 시각화할 수 있습니다. 스크립트가 끝날 때 wandb가 자동으로 마무리되므로 `.py` 스크립트를 실행할 때는 필요하지 않습니다.

노트북 환경에서 셸 명령어를 실행하려면 `!`를 앞에 붙여야 합니다. 예: `!wandb sync directoryname`.

### PyTorch 및 TensorBoard

PyTorch의 TensorBoard 통합을 사용하는 경우, 수동으로 PyTorch Profiler JSON 파일을 업로드해야 할 수도 있습니다**:** 

```
wandb.save(glob.glob(f"runs/*.pt.trace.json")[0], base_path=f"runs")
```