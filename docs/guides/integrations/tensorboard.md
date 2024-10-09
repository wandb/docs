---
title: TensorBoard
displayed_sidebar: default
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

## 코드 한 줄로 호스팅되는 TensorBoard

Weights & Biases를 사용하면 TensorBoard 로그를 클라우드에 쉽게 업로드할 수 있으며, 동료 및 클래스메이트와 결과를 빠르게 공유하고 분석을 하나의 중앙 위치에 보관할 수 있습니다.

<CTAButtons colabLink="https://github.com/wandb/examples/blob/master/colabs/tensorboard/TensorBoard_and_Weights_and_Biases.ipynb"></CTAButtons>

![](/images/integrations/tensorboard_oneline_code.webp)

### 코드 한 줄만 추가하세요

```python
import wandb

# `sync_tensorboard=True`로 wandb run 시작하기
wandb.init(project="my-project", sync_tensorboard=True)

# TensorBoard를 사용하는 트레이닝 코드
...

# [옵션] Notebook에서 실행하는 경우 tensorboard 로그를 W&B에 업로드하기 위해 wandb run을 종료합니다.
wandb.finish()
```

[**Weights & Biases에서 호스팅되는 Tensorboard 예제는 여기에서 확인하세요**](https://wandb.ai/rymc/simple-tensorboard-example/runs/oab614zf/tensorboard)

wandb run이 끝나면 TensorBoard 이벤트 파일이 Weights & Biases에 업로드됩니다. 이러한 메트릭은 **Weights & Biases 차트**에 기록되며 기계의 CPU 또는 GPU 활용도, git 상태, 사용된 터미널 코맨드 등 다양한 유용한 정보를 포함합니다.

:::info
Weights & Biases는 모든 버전의 TensorFlow에 대해 TensorBoard를 지원합니다. W&B는 PyTorch와 TensorBoardX가 포함된 TensorBoard > 1.14도 지원합니다.
:::

## 자주 묻는 질문

### TensorBoard에 로그되지 않은 메트릭을 W&B에 어떻게 로그할 수 있나요?

TensorBoard에 로그되지 않은 추가 사용자 정의 메트릭을 로그해야 하는 경우, 코드에서 `wandb.log`를 호출할 수 있습니다: `wandb.log({"custom": 0.8})`

Tensorboard를 동기화할 때는 `wandb.log`에서 step 인수를 설정할 수 없습니다. 다른 단계 수를 설정하고 싶다면, 다음과 같이 단계 메트릭과 함께 메트릭을 로그할 수 있습니다:

`wandb.log({"custom": 0.8, "global_step": global_step})`

### `wandb`와 함께 TensorBoard를 사용할 때 어떻게 설정하나요?

TensorBoard의 패치를 더 세밀하게 제어하려면 `wandb.init`에 `sync_tensorboard=True` 대신 `wandb.tensorboard.patch`를 호출할 수 있습니다.

```python
import wandb

wandb.tensorboard.patch(root_logdir="<logging_directory>")
wandb.init()

# Notebook에서 실행하는 경우 tensorboard 로그를 W&B에 업로드하기 위해 wandb run을 종료합니다.
wandb.finish()
```

이 메소드에 `tensorboard_x=False`를 전달하여 기본 TensorBoard가 패치되도록 할 수 있으며, PyTorch와 함께 TensorBoard > 1.14를 사용하는 경우 `pytorch=True`를 전달하여 패치되도록 할 수 있습니다. 이러한 옵션들은 가져온 라이브러리의 버전에 따라 자동으로 스마트한 기본값을 가집니다.

기본적으로, `tfevents` 파일과 모든 `.pbtxt` 파일도 동기화됩니다. 이를 통해 사용자를 대신해 TensorBoard 인스턴스를 시작할 수 있습니다. 실행 페이지에서 [TensorBoard 탭](https://www.wandb.com/articles/hosted-tensorboard)을 볼 수 있습니다. 이 행동은 `wandb.tensorboard.patch`에 `save=False`를 전달하여 비활성화할 수 있습니다.

```python
import wandb

wandb.init()
wandb.tensorboard.patch(save=False, tensorboard_x=True)

# 노트북에서 실행 중인 경우, wandb run을 종료하여 tensorboard 로그를 W&B에 업로드합니다.
wandb.finish()
```

:::caution
`tf.summary.create_file_writer`를 호출하거나 `torch.utils.tensorboard`를 통해 `SummaryWriter`를 생성하기 **전에** `wandb.init` 또는 `wandb.tensorboard.patch`를 호출해야 합니다.
:::

### 이전 TensorBoard Runs 동기화하기

로컬에 저장된 기존 `tfevents` 파일이 있고 이를 W&B로 가져오고 싶다면 `wandb sync log_dir`을 실행할 수 있습니다. 여기서 `log_dir`은 `tfevents` 파일이 포함된 로컬 디렉토리입니다.

### Google Colab, Jupyter 및 TensorBoard

Jupyter나 Colab 노트북에서 코드를 실행하는 경우, 트레이닝의 끝에서 `wandb.finish()`를 호출해야 합니다. 이는 wandb run을 종료하고 tensorboard 로그를 W&B에 업로드하여 시각화를 가능하게 합니다. `.py` 스크립트를 실행할 때는 wandb가 스크립트가 끝날 때 자동으로 종료되므로 필요하지 않습니다.

노트북 환경에서 셸 코맨드를 실행하려면 앞에 `!`를 추가해야 합니다, 예: `!wandb sync directoryname`.

### PyTorch와 TensorBoard

PyTorch의 TensorBoard 인테그레이션을 사용하는 경우, PyTorch Profiler JSON 파일을 수동으로 업로드해야 할 수도 있습니다**:**

```
wandb.save(glob.glob(f"runs/*.pt.trace.json")[0], base_path=f"runs")
```