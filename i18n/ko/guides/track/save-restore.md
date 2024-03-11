---
description: Save files to the cloud and restore them locally later
displayed_sidebar: default
---

# 파일 저장 및 복원

<head>
  <title>클라우드에 파일 저장 및 복원</title>
</head>

이 가이드는 먼저 `wandb.save`를 사용하여 클라우드에 파일을 저장하는 방법을 보여준 다음, `wandb.restore`를 사용하여 로컬에서 파일을 다시 생성하는 방법을 보여줍니다.

## 파일 저장하기

때때로, 숫자 값이나 미디어 조각을 로깅하는 것보다, 전체 파일을 로깅하고 싶을 때가 있습니다: 모델의 가중치, 다른 로깅 소프트웨어의 출력, 심지어 소스 코드까지도요.

W&B와 실행을 연결하고 클라우드에 업로드하는 데는 두 가지 방법이 있습니다.

1. `wandb.save(filename)`을 사용하세요.
2. 파일을 wandb 실행 디렉토리에 넣으면 실행이 끝날 때 업로드됩니다.

:::info
[다시 시작](../runs/resuming.md)하는 실행을 하고 있다면, `wandb.restore(filename)`을 호출하여 파일을 복구할 수 있습니다.
:::

파일이 작성되는 동안 파일을 동기화하려면, `wandb.save`에서 파일 이름 또는 글로브를 지정할 수 있습니다.

### `wandb.save`의 예시들

완전한 작동 예제는 [이 리포트](https://app.wandb.ai/lavanyashukla/save\_and\_restore/reports/Saving-and-Restoring-Models-with-W%26B--Vmlldzo3MDQ3Mw)를 참조하세요.

```python
# 현재 디렉토리에서 모델 파일 저장
wandb.save("model.h5")

# "ckpt"를 포함하는 모든 파일 저장
wandb.save("../logs/*ckpt*")

# "checkpoint"로 시작하는 파일들을 작성됨에 따라 저장
wandb.save(os.path.join(wandb.run.dir, "checkpoint*"))
```

:::info
W&B의 로컬 실행 디렉토리는 기본적으로 스크립트와 상대적인 `./wandb` 디렉토리 내에 있으며, 경로는 `run-20171023_105053-3o4933r0`와 같은 형식입니다. 여기서 `20171023_105053`는 타임스탬프이고, `3o4933r0`는 실행 ID입니다. `WANDB_DIR` [환경 변수](environment-variables.md)를 설정하거나, [`wandb.init`](./launch.md)의 `dir` 키워드 인수를 절대 경로로 설정하면, 파일이 해당 디렉토리 내에 작성됩니다.
:::

### 저장 정책과 상대 경로

`wandb.save`는 기본적으로 "**live**"로 설정된 **policy** 인수를 받습니다. 사용 가능한 정책은:

* **live (기본)** - 이 파일을 즉시 wandb 서버에 동기화하고 변경될 경우 다시 동기화
* **now** - 이 파일을 즉시 wandb 서버에 동기화하고, 변경되어도 계속 동기화하지 않음
* **end** - 실행이 끝날 때만 파일 동기화

또한, `wandb.save`에 **base\_path** 인수를 지정할 수 있습니다. 이를 통해 디렉토리 계층을 유지할 수 있습니다. 예를 들어:

```python
wandb.save(path="./results/eval/*", base_path="./results", policy="now")
```

이렇게 하면, 패턴에 일치하는 모든 파일이 루트가 아닌 `eval` 폴더에 저장됩니다.

:::info
`wandb.save`가 호출되면 제공된 경로에 존재하는 모든 파일을 나열하고 실행 디렉토리(`wandb.run.dir`)로 이들에 대한 심볼릭 링크를 생성합니다. `wandb.save`를 호출한 후 같은 경로에 새로운 파일을 생성하면 동기화하지 않습니다. 파일을 `wandb.run.dir`에 직접 작성하거나 새로운 파일이 생성될 때마다 `wandb.save`를 반드시 호출해야 합니다.
:::

### wandb 실행 디렉토리에 파일 저장 예시

파일 `model.h5`는 `wandb.run.dir`에 저장되며, 트레이닝이 끝날 때 업로드됩니다.

```python
import wandb

wandb.init()

model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    callbacks=[wandb.keras.WandbCallback()],
)
model.save(os.path.join(wandb.run.dir, "model.h5"))
```

여기 공개 예제 페이지가 있습니다. 파일 탭에서 `model-best.h5`를 볼 수 있습니다. 이것은 Keras 통합에 의해 기본적으로 자동 저장되지만, 수동으로 체크포인트를 저장하면 실행과 연결하여 저장합니다.

[실제 예제 보기 →](https://app.wandb.ai/wandb/neurips-demo/runs/206aacqo/files)

![](/images/experiments/example_saving_file_to_directory.png)

## 파일 복원하기

`wandb.restore(filename)`을 호출하면 로컬 실행 디렉토리에 파일을 복원합니다. 일반적으로 `filename`은 이전 실험 실행에 의해 생성되고 `wandb.save`를 사용하여 우리 클라우드에 업로드된 파일을 가리킵니다. 이 호출은 파일의 로컬 복사본을 만들고 읽기용으로 열린 로컬 파일 스트림을 반환합니다.

자주 사용되는 경우:

* 과거 실행에 의해 생성된 모델 아키텍처 또는 가중치 복원(더 복잡한 버전 관리 유스 케이스의 경우, [Artifacts](../artifacts/intro.md)를 참조하세요).
* 실패한 경우 마지막 체크포인트에서 트레이닝을 이어감([다시 시작](../runs/resuming.md) 섹션에 중요한 세부 사항이 있습니다)

### `wandb.restore`의 예시들

완전한 작동 예제는 [이 리포트](https://app.wandb.ai/lavanyashukla/save\_and\_restore/reports/Saving-and-Restoring-Models-with-W%26B--Vmlldzo3MDQ3Mw)를 참조하세요.

```python
# "vanpelt" 사용자의 "my-project"에서 특정 실행으로부터 모델 파일 복원
best_model = wandb.restore("model-best.h5", run_path="vanpelt/my-project/a1b2c3d")

# 체크포인트에서 가중치 파일 복원
# (참고: 실행 경로가 제공되지 않는 경우 다시 시작을 구성해야 함)
weights_file = wandb.restore("weights.h5")
# 반환된 오브젝트의 "name" 속성을 사용하세요
# 프레임워크가 파일 이름을 예상하는 경우에 사용하십시오, 예를 들어 Keras에서처럼
my_predefined_model.load_weights(weights_file.name)
```

> `run_path`를 지정하지 않으면, 실행에 대해 [다시 시작](../runs/resuming.md)을 구성해야 합니다. 트레이닝 외부에서 프로그래매틱하게 파일에 엑세스하려면 [Run API](../../ref/python/run.md)를 사용하세요.

## 자주 묻는 질문

### 어떤 파일들을 무시하나요?

`wandb/settings` 파일을 편집하고 `ignore_globs`를 쉼표로 구분된 [글로브](https://en.wikipedia.org/wiki/Glob\_\(programming\)) 목록과 같게 설정할 수 있습니다. `WANDB_IGNORE_GLOBS` [환경 변수](./environment-variables.md)도 설정할 수 있습니다. 자동으로 생성되는 git 패치가 업로드되는 것을 방지하는 것이 일반적인 사용 사례입니다. 예를 들어 `WANDB_IGNORE_GLOBS=*.patch`.

### 파일 저장 디렉토리 변경하기

AWS S3 또는 Google Cloud Storage에서 파일을 기본적으로 저장하는 경우 다음 오류가 발생할 수 있습니다:`events.out.tfevents.1581193870.gpt-tpu-finetune-8jzqk-2033426287은 클라우드 스토리지 URL입니다, wandb에 파일을 저장할 수 없습니다.`

TensorBoard 이벤트 파일 또는 동기화하고자 하는 기타 파일의 로그 디렉토리를 변경하려면, 파일을 `wandb.run.dir`에 저장하여 우리 클라우드에 동기화하세요.

### 실행 이름을 어떻게 얻나요?

스크립트 내에서 실행 이름을 사용하고 싶다면, `wandb.run.name`을 사용하고 예를 들어 "blissful-waterfall-2"와 같은 실행 이름을 얻을 수 있습니다.

디스플레이 이름에 접근하기 전에 실행을 저장해야 합니다:

```
run = wandb.init(...)
run.save()
print(run.name)
```

### 로컬에 저장된 모든 파일을 어떻게 푸시하나요?

`wandb.init` 후 스크립트 상단에서 한 번 `wandb.save("*.pt")`를 호출하면, `wandb.run.dir`에 작성된 모든 파일이 즉시 저장됩니다.

### 이미 클라우드 스토리지에 동기화된 로컬 파일을 제거할 수 있나요?

`wandb sync --clean` 코맨드를 실행하여 이미 클라우드 스토리지에 동기화된 로컬 파일을 제거할 수 있습니다. 사용법에 대한 자세한 정보는 `wandb sync --help`로 찾을 수 있습니다.

### 내 코드의 상태를 복원하고 싶다면 어떻게 하나요?

[커맨드라인 툴](../../ref/cli/README.md)의 `restore` 코맨드를 사용하여 주어진 실행을 수행했을 때의 코드 상태로 돌아갑니다.

```shell
# 브랜치를 생성하고 실행 $RUN_ID가 실행됐을 때의
# 코드 상태로 복원합니다
wandb restore $RUN_ID
```

### `wandb`는 어떻게 코드의 상태를 캡쳐하나요?

스크립트에서 `wandb.init`이 호출될 때, 코드가 git 저장소에 있으면 마지막 git 커밋에 대한 링크가 저장됩니다. 변경사항이 커밋되지 않았거나 원격과 동기화되지 않은 경우에 대비하여 diff 패치도 생성됩니다.