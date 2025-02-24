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
W&B 는 W&B Multi-tenant SaaS 용 임베디드 TensorBoard 를 지원합니다.
{{% /alert %}}

TensorBoard 로그를 클라우드에 업로드하고, 동료 및 급우들과 결과를 빠르게 공유하고, 분석 을 한 곳에서 중앙 집중식으로 관리하세요.

{{< img src="/images/integrations/tensorboard_oneline_code.webp" alt="" >}}

## 시작하기

```python
import wandb

# `sync_tensorboard=True` 로 wandb run 시작
wandb.init(project="my-project", sync_tensorboard=True)

# TensorBoard 를 사용한 트레이닝 코드
...

# [선택 사항] TensorBoard 로그를 W&B 에 업로드하기 위해 wandb run 을 종료합니다 (노트북에서 실행하는 경우).
wandb.finish()
```

[예제](https://wandb.ai/rymc/simple-tensorboard-example/runs/oab614zf/tensorboard) 를 검토하세요.

run 이 완료되면 W&B 에서 TensorBoard 이벤트 파일에 엑세스할 수 있으며 시스템의 CPU 또는 GPU 사용률, `git` 상태, run 에서 사용된 터미널 코맨드 등과 같은 추가 유용한 정보와 함께 기본 W&B 차트에서 메트릭 을 시각화할 수 있습니다.

{{% alert %}}
W&B 는 모든 버전의 TensorFlow 에서 TensorBoard 를 지원합니다. 또한 W&B 는 PyTorch 및 TensorBoardX 와 함께 TensorBoard 1.14 이상을 지원합니다.
{{% /alert %}}

## 자주 묻는 질문

### TensorBoard 에 기록되지 않은 메트릭 을 W&B 에 어떻게 기록할 수 있습니까?

TensorBoard 에 기록되지 않은 추가 사용자 지정 메트릭 을 기록해야 하는 경우 코드에서 `wandb.log` 를 호출할 수 있습니다. `wandb.log({"custom": 0.8})`

Tensorboard 를 동기화할 때 `wandb.log` 에서 step 인수를 설정하는 기능은 꺼집니다. 다른 step 카운트를 설정하려면 다음과 같이 step 메트릭 으로 메트릭 을 기록할 수 있습니다.

`wandb.log({"custom": 0.8, "global_step": global_step})`

### `wandb` 와 함께 사용할 때 Tensorboard 를 어떻게 구성합니까?

TensorBoard 를 패치하는 방법을 더 세밀하게 제어하려면 `wandb.init` 에 `sync_tensorboard=True` 를 전달하는 대신 `wandb.tensorboard.patch` 를 호출할 수 있습니다.

```python
import wandb

wandb.tensorboard.patch(root_logdir="<logging_directory>")
wandb.init()

# TensorBoard 로그를 W&B 에 업로드하기 위해 wandb run 을 종료합니다 (노트북에서 실행하는 경우).
wandb.finish()
```

TensorBoard > 1.14 를 PyTorch 와 함께 사용하는 경우 `tensorboard_x=False` 를 이 메소드에 전달하여 일반 TensorBoard 가 패치되었는지 확인할 수 있으며, `pytorch=True` 를 전달하여 패치되었는지 확인할 수 있습니다. 이러한 옵션은 모두 이러한 라이브러리의 어떤 버전이 임포트되었는지에 따라 스마트 기본값을 갖습니다.

기본적으로 `tfevents` 파일과 모든 `.pbtxt` 파일도 동기화합니다. 이를 통해 사용자를 대신하여 TensorBoard 인스턴스를 시작할 수 있습니다. run 페이지에 [TensorBoard 탭](https://www.wandb.com/articles/hosted-tensorboard)이 표시됩니다. 이 동작은 `wandb.tensorboard.patch` 에 `save=False` 를 전달하여 끌 수 있습니다.

```python
import wandb

wandb.init()
wandb.tensorboard.patch(save=False, tensorboard_x=True)

# 노트북에서 실행하는 경우 TensorBoard 로그를 W&B 에 업로드하기 위해 wandb run 을 종료합니다.
wandb.finish()
```

{{% alert color="secondary" %}}
`tf.summary.create_file_writer` 를 호출하거나 `torch.utils.tensorboard` 를 통해 `SummaryWriter` 를 구성하기 **전에** `wandb.init` 또는 `wandb.tensorboard.patch` 를 호출해야 합니다.
{{% /alert %}}

### 이전 TensorBoard run 을 어떻게 동기화합니까?

로컬에 저장된 기존 `tfevents` 파일이 있고 이를 W&B 로 임포트하려면 `wandb sync log_dir` 를 실행합니다. 여기서 `log_dir` 는 `tfevents` 파일이 포함된 로컬 디렉토리입니다.

### Google Colab 또는 Jupyter 를 TensorBoard 와 함께 어떻게 사용합니까?

Jupyter 또는 Colab 노트북에서 코드를 실행하는 경우 트레이닝 이 끝나면 반드시 `wandb.finish()` 를 호출하십시오. 이렇게 하면 wandb run 이 완료되고 TensorBoard 로그가 W&B 에 업로드되어 시각화할 수 있습니다. wandb 는 스크립트가 완료되면 자동으로 완료되므로 `.py` 스크립트 를 실행할 때는 필요하지 않습니다.

노트북 환경에서 셸 코맨드 를 실행하려면 `!wandb sync directoryname` 와 같이 `!` 를 앞에 붙여야 합니다.

### PyTorch 를 TensorBoard 와 함께 어떻게 사용합니까?

PyTorch 의 TensorBoard 인테그레이션 을 사용하는 경우 PyTorch Profiler JSON 파일을 수동으로 업로드해야 할 수 있습니다.

```python
wandb.save(glob.glob(f"runs/*.pt.trace.json")[0], base_path=f"runs")
```
