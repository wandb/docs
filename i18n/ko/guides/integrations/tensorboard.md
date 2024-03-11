---
displayed_sidebar: default
---

# TensorBoard

## 1줄의 코드로 호스팅되는 TensorBoard

Weight & Biases를 사용하면 TensorBoard 로그를 클라우드에 쉽게 업로드하고, 결과를 동료나 동급생과 빠르게 공유하며, 분석을 한 곳에 중앙 집중화할 수 있습니다.

**이 노트북으로 지금 시작하세요:** [**Colab 노트북에서 시도해보기 →**](https://github.com/wandb/examples/blob/master/colabs/tensorboard/TensorBoard\_and\_Weights\_and\_Biases.ipynb)


![](/images/integrations/tensorboard_oneline_code.webp)

### 단 1줄의 코드만 추가하세요

```python
import wandb

# `sync_tensorboard=True`를 사용하여 wandb run을 시작합니다
wandb.init(project="my-project", sync_tensorboard=True)

# TensorBoard를 사용하는 귀하의 트레이닝 코드
...

# [선택사항] wandb run을 마쳐서 TensorBoard 로그를 W&B에 업로드합니다(노트북에서 실행하는 경우)
wandb.finish()
```

[**Weights & Biases에서 호스팅되는 Tensorboard의 예시 보기**](https://wandb.ai/rymc/simple-tensorboard-example/runs/oab614zf/tensorboard)

wandb run이 완료되면, 귀하의 TensorBoard 이벤트 파일은 Weights & Biases에 업로드됩니다. 이 메트릭은 **또한** 귀하의 기계의 CPU 또는 GPU 사용량, git 상태, 사용된 터미널 코맨드 등과 같이 유용한 정보와 함께 네이티브 Weights & Biases 차트에도 기록됩니다.

:::info
Weights & Biases는 모든 버전의 TensorFlow와 함께 TensorBoard를 지원합니다. W&B는 또한 PyTorch와 함께 1.14 이상의 TensorBoard와 TensorBoardX도 지원합니다.
:::

## 자주 묻는 질문

### TensorBoard에 기록되지 않는 메트릭을 W&B에 어떻게 기록하나요?

TensorBoard에 기록되지 않는 추가적인 사용자 정의 메트릭을 기록해야 하는 경우, 코드에서 `wandb.log`를 호출할 수 있습니다 `wandb.log({"custom": 0.8})`

`wandb.log`에서 step 인수를 설정하는 것은 Tensorboard와 동기화할 때 비활성화됩니다. 다른 step 수를 설정하고 싶다면, step 메트릭과 함께 메트릭을 기록할 수 있습니다:

`wandb.log({"custom": 0.8, "global_step": global_step})`

### `wandb`를 사용할 때 Tensorboard를 어떻게 구성하나요?

TensorBoard가 패치되는 방식을 더 많이 제어하고 싶다면, `wandb.init`에 `sync_tensorboard=True`를 전달하는 대신 `wandb.tensorboard.patch`를 호출할 수 있습니다.

```python
import wandb

wandb.tensorboard.patch(root_logdir="<logging_directory>")
wandb.init()

# W&B에 tensorboard 로그를 업로드하기 위해 wandb run을 마칩니다(노트북에서 실행하는 경우)
wandb.finish()
```

이 메소드에 `tensorboard_x=False`를 전달하여 PyTorch를 사용하여 1.14 이상의 TensorBoard를 사용하는 경우에는 기본 TensorBoard가 패치되도록 할 수 있습니다. `pytorch=True`를 전달하여 패치되도록 할 수도 있습니다. 이러한 옵션은 모두 가져온 이러한 라이브러리의 버전에 따라 스마트한 기본값을 가집니다.

기본적으로, 우리는 `tfevents` 파일과 모든 `.pbtxt` 파일도 동기화합니다. 이를 통해 우리는 귀하를 대신하여 TensorBoard 인스턴스를 시작할 수 있습니다. run 페이지에서 [TensorBoard 탭](https://www.wandb.com/articles/hosted-tensorboard)을 볼 수 있습니다. 이 동작은 `wandb.tensorboard.patch`에 `save=False`를 전달하여 비활성화할 수 있습니다.

```python
import wandb

wandb.init()
wandb.tensorboard.patch(save=False, tensorboard_x=True)

# 노트북에서 실행하는 경우, wandb run을 마쳐서 W&B에 tensorboard 로그를 업로드합니다
wandb.finish()
```

:::caution
`tf.summary.create_file_writer`를 호출하거나 `torch.utils.tensorboard`를 통해 `SummaryWriter`를 구성하기 **전에** 반드시 `wandb.init` 또는 `wandb.tensorboard.patch`를 호출해야 합니다.
:::

### 이전 TensorBoard Runs 동기화

로컬에 저장된 기존의 `tfevents` 파일을 W&B로 가져오고 싶다면, `log_dir`이 `tfevents` 파일을 포함하고 있는 로컬 디렉토리인 `wandb sync log_dir`을 실행할 수 있습니다.

### Google Colab, Jupyter 및 TensorBoard

Jupyter 또는 Colab 노트북에서 코드를 실행하는 경우, 트레이닝이 끝난 후에 `wandb.finish()`를 호출해야 합니다. 이는 wandb run을 마치고 tensorboard 로그를 W&B에 업로드하여 시각화할 수 있게 합니다. `.py` 스크립트를 실행하는 경우에는 스크립트가 완료될 때 자동으로 wandb가 종료되므로 필요하지 않습니다.

노트북 환경에서 셸 코맨드를 실행하려면, `!`를 앞에 붙여야 합니다. 예를 들어 `!wandb sync directoryname`.

### PyTorch 및 TensorBoard

PyTorch의 TensorBoard 통합을 사용하는 경우, PyTorch 프로파일러 JSON 파일을 수동으로 업로드해야 할 수도 있습니다**:** 

```
wandb.save(glob.glob(f"runs/*.pt.trace.json")[0], base_path=f"runs")
```