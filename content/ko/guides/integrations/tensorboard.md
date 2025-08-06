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
W&B는 W&B Multi-tenant SaaS에서 임베디드 TensorBoard를 지원합니다.
{{% /alert %}}

TensorBoard 로그를 클라우드에 업로드하여, 동료나 학우들과 빠르게 결과를 공유하고 모든 분석 자료를 한 곳에서 체계적으로 관리하세요.

{{< img src="/images/integrations/tensorboard_oneline_code.webp" alt="TensorBoard 인테그레이션 코드" >}}

## 시작하기

```python
import wandb

# `sync_tensorboard=True` 옵션으로 wandb run을 시작하세요
wandb.init(project="my-project", sync_tensorboard=True) as run:
  # TensorBoard를 사용하는 트레이닝 코드
  ...

```

[TensorBoard 인테그레이션 예시 run](https://wandb.ai/rymc/simple-tensorboard-example/runs/oab614zf/tensorboard)을 확인해보세요.

run이 종료되면, W&B에서 TensorBoard 이벤트 파일에 엑세스할 수 있고, 메트릭을 W&B의 기본 차트에서 시각화할 수 있습니다. 시스템의 CPU 또는 GPU 사용량, `git` 상태, run에 사용된 터미널 코맨드 등 유용한 추가 정보도 함께 볼 수 있습니다.

{{% alert %}}
W&B는 모든 버전의 TensorFlow와 함께 TensorBoard를 지원합니다. 또한 W&B는 PyTorch 및 TensorBoardX와 함께 TensorBoard 1.14 이상 버전도 지원합니다.
{{% /alert %}}

## 자주 묻는 질문

### TensorBoard에 기록되지 않은 메트릭도 W&B에 로그할 수 있나요?

TensorBoard에 기록되지 않는 커스텀 메트릭을 추가로 로그하고 싶다면 코드에서 `wandb.Run.log()`를 호출하면 됩니다. 예시: `run.log({"custom": 0.8})`

TensorBoard와 연동 중에는 `run.log()`의 step 인수 사용이 비활성화됩니다. 다른 step 값을 직접 지정하고 싶다면 아래와 같이 step 메트릭과 함께 기록하세요.

`run.log({"custom": 0.8, "global_step": global_step})`

### `wandb`와 함께 TensorBoard를 사용할 때 설정을 직접 조정하려면 어떻게 하나요?

TensorBoard를 패치하는 방법을 더 세밀하게 제어하고 싶다면, `wandb.init`에 `sync_tensorboard=True`를 넘기는 대신 `wandb.tensorboard.patch`를 호출할 수 있습니다.

```python
import wandb

wandb.tensorboard.patch(root_logdir="<logging_directory>")
run = wandb.init()

# 노트북 환경인 경우 run을 종료하여 tensorboard 로그를 업로드하세요
run.finish()
```

`tensorboard_x=False`를 넘기면 기본 TensorBoard만 패치되고, TensorBoard > 1.14에서 PyTorch를 사용하는 경우 `pytorch=True`를 넘겨 패치할 수 있습니다. 각 옵션은 가져온 라이브러리 버전에 따라 자동으로 동작합니다.

기본적으로 `tfevents` 파일과 `.pbtxt` 파일도 함께 동기화됩니다. 이를 통해 여러분을 대신해 TensorBoard 인스턴스를 띄울 수 있으며, run 페이지에서 [TensorBoard 탭](https://www.wandb.com/articles/hosted-tensorboard)을 확인할 수 있습니다. 이 동작은 `wandb.tensorboard.patch`에 `save=False`를 넘겨서 비활성화할 수 있습니다.

```python
import wandb

run = wandb.init()
wandb.tensorboard.patch(save=False, tensorboard_x=True)

# 노트북 환경인 경우 run을 종료하여 tensorboard 로그를 업로드하세요
run.finish()
```

{{% alert color="secondary" %}}
반드시 `tf.summary.create_file_writer` 호출이나 `torch.utils.tensorboard`로 `SummaryWriter`를 생성하기 전에 `wandb.init()` 또는 `wandb.tensorboard.patch`를 먼저 호출해야 합니다.
{{% /alert %}}

### 기존 TensorBoard run은 어떻게 동기화하나요?

로컬에 저장된 기존 `tfevents` 파일을 W&B로 가져오려면, 해당 파일이 있는 디렉토리에서 `wandb sync log_dir`을 실행하세요. 여기서 `log_dir`은 `tfevents` 파일이 들어있는 로컬 디렉토리입니다.

### Google Colab이나 Jupyter에서 TensorBoard를 어떻게 사용하나요?

Jupyter 또는 Colab 노트북에서 코드를 실행할 경우, 트레이닝 마지막에 `wandb.Run.finish()`를 꼭 호출해야 합니다. 이 명령어는 wandb run을 종료하고 tensorboard 로그를 W&B에 업로드해 주어 시각화가 가능합니다. `.py` 스크립트로 실행할 때는 스크립트가 끝나면 자동으로 wandb가 종료되므로 따로 호출할 필요가 없습니다.

노트북 환경에서 셸 코맨드를 실행하려면 명령어 앞에 `!`를 붙여야 합니다. 예시: `!wandb sync directoryname`

### PyTorch와 TensorBoard를 함께 쓰려면 어떻게 하나요?

PyTorch의 TensorBoard 인테그레이션을 사용할 경우, PyTorch Profiler에서 생성된 JSON 파일을 수동으로 업로드해야 할 수 있습니다.

```python
with wandb.init(project="my-project", sync_tensorboard=True) as run:
    run.save(glob.glob(f"runs/*.pt.trace.json")[0], base_path=f"runs")
```

### 클라우드에 저장된 tfevents 파일도 동기화할 수 있나요?

`wandb` 0.20.0 이상 버전에서는 S3, GCS, Azure에 저장된 `tfevents` 파일 동기화가 지원됩니다. 각 클라우드 제공업체의 기본 인증 정보를 사용하며, 아래 표의 명령어와 포맷을 참고하세요.

| 클라우드 제공업체 | 인증 명령어                                 | 로그 디렉토리 형식                      |
| ---------------- | ------------------------------------------ | -------------------------------------- |
| S3               | `aws configure`                            | `s3://bucket/path/to/logs`             |
| GCS              | `gcloud auth application-default login`    | `gs://bucket/path/to/logs`             |
| Azure            | `az login`[^1]                             | `az://account/container/path/to/logs`  |

[^1]: `AZURE_STORAGE_ACCOUNT`와 `AZURE_STORAGE_KEY` 환경 변수를 추가로 설정해야 합니다.