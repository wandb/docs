---
title: TensorBoard
menu:
  default:
    identifier: ko-guides-integrations-tensorboard
    parent: integrations
weight: 430
---

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/tensorboard/TensorBoard_and_Weights_and_Biases.ipynb" >}}

{{% alert %}}
W&B는 W&B 멀티 테넌트 SaaS용 임베디드 TensorBoard를 지원합니다.
{{% /alert %}}

TensorBoard 로그를 클라우드에 업로드하고, 동료 및 급우들 사이에서 결과를 빠르게 공유하고, 분석을 한 곳에서 중앙 집중식으로 관리하세요.

{{< img src="/images/integrations/tensorboard_oneline_code.webp" alt="" >}}

## 시작하기

```python
import wandb

# `sync_tensorboard=True`로 wandb run을 시작합니다.
wandb.init(project="my-project", sync_tensorboard=True)

# TensorBoard를 사용하는 트레이닝 코드
...

# [선택 사항] W&B에 tensorboard 로그를 업로드하기 위해 wandb run을 완료합니다(노트북에서 실행하는 경우).
wandb.finish()
```

[예시](https://wandb.ai/rymc/simple-tensorboard-example/runs/oab614zf/tensorboard)를 검토하세요.

run이 완료되면 W&B에서 TensorBoard 이벤트 파일에 엑세스할 수 있으며 시스템의 CPU 또는 GPU 사용률, `git` 상태, run에서 사용한 터미널 코맨드 등과 같은 추가 유용한 정보와 함께 기본 W&B 차트에서 메트릭을 시각화할 수 있습니다.

{{% alert %}}
W&B는 모든 TensorFlow 버전에서 TensorBoard를 지원합니다. W&B는 PyTorch뿐만 아니라 TensorBoardX와 함께 TensorBoard 1.14 이상도 지원합니다.
{{% /alert %}}

## 자주 묻는 질문

### TensorBoard에 기록되지 않은 메트릭을 W&B에 어떻게 기록할 수 있습니까?

TensorBoard에 기록되지 않은 추가 사용자 정의 메트릭을 기록해야 하는 경우 코드에서 `wandb.log`를 호출할 수 있습니다. `wandb.log({"custom": 0.8})`

Tensorboard를 동기화할 때 `wandb.log`에서 step 인수를 설정하는 기능은 꺼져 있습니다. 다른 step 카운트를 설정하려면 다음과 같이 step 메트릭으로 메트릭을 기록할 수 있습니다.

`wandb.log({"custom": 0.8, "global_step": global_step})`

### `wandb`와 함께 Tensorboard를 사용할 때 Tensorboard를 어떻게 구성합니까?

TensorBoard 패치 방법을 보다 세밀하게 제어하려면 `wandb.init`에 `sync_tensorboard=True`를 전달하는 대신 `wandb.tensorboard.patch`를 호출할 수 있습니다.

```python
import wandb

wandb.tensorboard.patch(root_logdir="<logging_directory>")
wandb.init()

# W&B에 tensorboard 로그를 업로드하기 위해 wandb run을 완료합니다(노트북에서 실행하는 경우).
wandb.finish()
```

TensorBoard > 1.14를 PyTorch와 함께 사용하는 경우 vanilla TensorBoard가 패치되었는지 확인하려면 `tensorboard_x=False`를 이 메소드에 전달하고, 패치되었는지 확인하려면 `pytorch=True`를 전달할 수 있습니다. 이러한 옵션은 모두 이러한 라이브러리의 버전에 따라 스마트 기본값을 갖습니다.

기본적으로 `tfevents` 파일과 모든 `.pbtxt` 파일도 동기화합니다. 이를 통해 사용자를 대신하여 TensorBoard 인스턴스를 시작할 수 있습니다. run 페이지에 [TensorBoard 탭](https://www.wandb.com/articles/hosted-tensorboard)이 표시됩니다. 이 동작은 `wandb.tensorboard.patch`에 `save=False`를 전달하여 끌 수 있습니다.

```python
import wandb

wandb.init()
wandb.tensorboard.patch(save=False, tensorboard_x=True)

# 노트북에서 실행하는 경우 W&B에 tensorboard 로그를 업로드하기 위해 wandb run을 완료합니다.
wandb.finish()
```

{{% alert color="secondary" %}}
`tf.summary.create_file_writer`를 호출하거나 `torch.utils.tensorboard`를 통해 `SummaryWriter`를 구성하기 **전에** `wandb.init` 또는 `wandb.tensorboard.patch`를 호출해야 합니다.
{{% /alert %}}

### 이전 TensorBoard run을 어떻게 동기화합니까?

로컬에 저장된 기존 `tfevents` 파일이 있고 이를 W&B로 가져오려면 `wandb sync log_dir`을 실행합니다. 여기서 `log_dir`은 `tfevents` 파일이 포함된 로컬 디렉토리입니다.

### Google Colab 또는 Jupyter를 TensorBoard와 함께 어떻게 사용합니까?

Jupyter 또는 Colab 노트북에서 코드를 실행하는 경우 트레이닝이 끝나면 `wandb.finish()`를 호출해야 합니다. 이렇게 하면 wandb run이 완료되고 tensorboard 로그가 W&B에 업로드되어 시각화할 수 있습니다. `.py` 스크립트가 완료되면 wandb가 자동으로 완료되므로 이는 필요하지 않습니다.

노트북 환경에서 셸 코맨드를 실행하려면 `!wandb sync directoryname`과 같이 `!`를 앞에 붙여야 합니다.

### PyTorch를 TensorBoard와 함께 어떻게 사용합니까?

PyTorch의 TensorBoard 인테그레이션을 사용하는 경우 PyTorch Profiler JSON 파일을 수동으로 업로드해야 할 수 있습니다.

```python
wandb.save(glob.glob(f"runs/*.pt.trace.json")[0], base_path=f"runs")
```
