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

때로는 숫자 값이나 미디어 조각 대신 전체 파일을 로그하고 싶을 수 있습니다: 모델의 가중치, 다른 로깅 소프트웨어의 출력, 심지어 소스 코드까지도요.

파일을 실행과 연결하고 W&B에 업로드하는 두 가지 방법이 있습니다.

1. `wandb.save(filename)`을 사용합니다.
2. 파일을 wandb 실행 디렉터리에 넣으면 실행이 끝날 때 업로드됩니다.

:::info
실행을 [재개](../runs/resuming.md)하는 경우, `wandb.restore(filename)`을 호출하여 파일을 복구할 수 있습니다.
:::

파일이 작성되는 동안 동기화하려면 `wandb.save`에서 파일 이름이나 글로브를 지정할 수 있습니다.

### `wandb.save`의 예시

완전한 작동 예를 보려면 [이 리포트](https://app.wandb.ai/lavanyashukla/save\_and\_restore/reports/Saving-and-Restoring-Models-with-W%26B--Vmlldzo3MDQ3Mw)를 참조하세요.

```python
# 현재 디렉터리에서 모델 파일 저장
wandb.save("model.h5")

# "ckpt"라는 부분 문자열이 포함된 모든 파일 저장
wandb.save("../logs/*ckpt*")

# "checkpoint"로 시작하는 파일을 작성될 때 저장
wandb.save(os.path.join(wandb.run.dir, "checkpoint*"))
```

:::info
W&B의 로컬 실행 디렉터리는 기본적으로 스크립트와 관련하여 `./wandb` 디렉터리 내부에 있으며, 경로는 `20171023_105053`이 타임스탬프이고 `3o4933r0`이 실행 ID인 `run-20171023_105053-3o4933r0`처럼 보입니다. `WANDB_DIR` [환경 변수](environment-variables.md)를 설정하거나, [`wandb.init`](./launch.md)의 `dir` 키워드 인수를 절대 경로로 설정하면 해당 디렉터리 내에 파일이 작성됩니다.
:::

### 저장 정책 및 상대 경로

`wandb.save`는 기본적으로 "**live**"로 설정된 **policy** 인수를 받습니다. 사용 가능한 정책은 다음과 같습니다:

* **live (기본값)** - 이 파일을 즉시 wandb 서버에 동기화하고 변경될 경우 다시 동기화합니다
* **now** - 이 파일을 즉시 wandb 서버에 동기화하고, 변경되어도 계속 동기화하지 않습니다
* **end** - 실행이 끝날 때만 파일을 동기화합니다

또한 `wandb.save`에 **base\_path** 인수를 지정할 수 있습니다. 이를 통해 디렉터리 계층을 유지할 수 있습니다. 예를 들면:

```python
wandb.save(path="./results/eval/*", base_path="./results", policy="now")
```

이는 패턴과 일치하는 모든 파일이 루트가 아닌 `eval` 폴더에 저장됩니다.

:::info
`wandb.save`가 호출되면 제공된 경로에 존재하는 모든 파일을 나열하고 실행 디렉터리(`wandb.run.dir`)로 그들의 심볼릭 링크를 생성합니다. `wandb.save`를 호출한 후 같은 경로에 새 파일을 생성하는 경우, 우리는 그것들을 동기화하지 않습니다. 파일을 직접 `wandb.run.dir`에 작성하거나 새 파일이 생성될 때마다 `wandb.save`를 호출해야 합니다.
:::

### wandb 실행 디렉터리에 파일 저장 예시

파일 `model.h5`는 `wandb.run.dir`에 저장되며, 학습이 끝날 때 업로드됩니다.

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

여기에 공개 예시 페이지가 있습니다. 파일 탭에 `model-best.h5`가 있는 것을 볼 수 있습니다. 이것은 Keras 통합에 의해 기본적으로 자동 저장되지만, 수동으로 체크포인트를 저장할 수 있으며 실행과 연결하여 저장할 것입니다.

[실시간 예제 보기 →](https://app.wandb.ai/wandb/neurips-demo/runs/206aacqo/files)

![](/images/experiments/example_saving_file_to_directory.png)

## 파일 복원하기

`wandb.restore(filename)`를 호출하면 로컬 실행 디렉터리에 파일을 복원합니다. 일반적으로 `filename`은 이전 실험 실행에 의해 생성되고 `wandb.save`로 클라우드에 업로드된 파일을 나타냅니다. 이 호출은 파일의 로컬 복사본을 만들고 읽기용으로 열린 로컬 파일 스트림을 반환합니다.

일반적인 사용 사례:

* 과거 실행에 의해 생성된 모델 아키텍처 또는 가중치 복원 (더 복잡한 버전 관리 사용 사례의 경우, [아티팩트](../artifacts/intro.md)를 참조하세요.
* 실패한 경우 마지막 체크포인트부터 학습을 재개 (중요한 세부 사항에 대해서는 [재개](../runs/resuming.md) 섹션을 참조하세요)

### `wandb.restore`의 예시

완전한 작동 예를 보려면 [이 리포트](https://app.wandb.ai/lavanyashukla/save\_and\_restore/reports/Saving-and-Restoring-Models-with-W%26B--Vmlldzo3MDQ3Mw)를 참조하세요.

```python
# "my-project"의 사용자 "vanpelt"에 의한 특정 실행에서 모델 파일 복원
best_model = wandb.restore("model-best.h5", run_path="vanpelt/my-project/a1b2c3d")

# 체크포인트에서 가중치 파일 복원
# (참고: 실행 경로가 제공되지 않는 경우 재개가 구성되어야 함)
weights_file = wandb.restore("weights.h5")
# 프레임워크가 파일 이름을 예상하는 경우 반환된 객체의 "name" 속성을 사용하세요. 예를 들어 Keras에서는
my_predefined_model.load_weights(weights_file.name)
```

> `run_path`를 지정하지 않으면, 실행을 위해 [재개](../runs/resuming.md)를 구성해야 합니다. 훈련 외부에서 프로그래매틱하게 파일에 액세스하려면 [Run API](../../ref/python/run.md)를 사용하세요.

## 자주 묻는 질문

### 파일을 무시하려면 어떻게 하나요?

`wandb/settings` 파일을 편집하고 `ignore_globs`를 쉼표로 구분된 [글로브](https://en.wikipedia.org/wiki/Glob\_\(programming\)) 목록으로 설정할 수 있습니다. 또한 `WANDB_IGNORE_GLOBS` [환경 변수](./environment-variables.md)를 설정할 수 있습니다. 일반적인 사용 사례는 우리가 자동으로 생성하는 git 패치가 업로드되는 것을 방지하는 것입니다. 즉, `WANDB_IGNORE_GLOBS=*.patch`.

### 파일 저장 디렉터리 변경하기

AWS S3 또는 Google Cloud Storage에 파일을 저장하는 것이 기본 설정인 경우, 다음과 같은 오류가 발생할 수 있습니다: `events.out.tfevents.1581193870.gpt-tpu-finetune-8jzqk-2033426287은 클라우드 저장소 url이므로, wandb에 파일을 저장할 수 없습니다.`

TensorBoard 이벤트 파일이나 동기화하고 싶은 다른 파일의 로그 디렉터리를 변경하려면, `wandb.run.dir`에 파일을 저장하여 우리 클라우드에 동기화되도록 하세요.

### 실행 이름을 어떻게 얻나요?

스크립트 내에서 실행 이름을 사용하고 싶다면, `wandb.run.name`을 사용하면 됩니다. 예를 들어 "blissful-waterfall-2"와 같은 실행 이름을 얻을 수 있습니다.

디스플레이 이름에 액세스하기 전에 실행을 저장해야 합니다:

```
run = wandb.init(...)
run.save()
print(run.name)
```

### 로컬에 저장된 모든 파일을 어떻게 푸시하나요?

`wandb.init` 후 스크립트 상단에서 한 번 `wandb.save("*.pt")`를 호출하면, `wandb.run.dir`에 작성된 모든 파일이 즉시 저장됩니다.

### 이미 클라우드 스토리지에 동기화된 로컬 파일을 제거할 수 있나요?

`wandb sync --clean` 명령을 실행하여 이미 클라우드 스토리지에 동기화된 로컬 파일을 제거할 수 있습니다. 사용법에 대한 자세한 정보는 `wandb sync --help`로 찾을 수 있습니다.

### 코드의 상태를 복원하고 싶다면?

[명령 줄 도구](../../ref/cli/README.md)의 `restore` 명령을 사용하여 특정 실행을 수행했을 때의 코드 상태로 돌아갈 수 있습니다.

```shell
# 브랜치를 생성하고 실행 $RUN_ID가 실행될 때의
# 코드 상태로 복원합니다
wandb restore $RUN_ID
```

### `wandb`는 코드의 상태를 어떻게 캡처하나요?

`wandb.init`이 스크립트에서 호출될 때, 코드가 git 저장소에 있으면 마지막 git 커밋에 대한 링크가 저장됩니다. 또한 커밋되지 않았거나 원격과 동기화되지 않은 변경 사항이 있는 경우 차이 패치도 생성됩니다.