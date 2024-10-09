---
title: Save & restore files to the cloud
description: 클라우드에 파일을 저장하고 나중에 로컬에서 복원하세요
displayed_sidebar: default
---

이 가이드는 먼저 `wandb.save`를 사용하여 파일을 클라우드에 저장하는 방법을 보여주고, 그런 다음 `wandb.restore`로 로컬에서 재생성할 수 있는 방법을 보여줍니다.

## 파일 저장

때때로, 숫자 값이나 미디어 조각을 로깅하는 대신, 전체 파일을 로깅하고 싶을 때가 있습니다: 모델의 가중치, 다른 로깅 소프트웨어의 출력, 소스 코드까지도.

파일을 run에 연결하고 W&B에 업로드하는 두 가지 방법이 있습니다.

1. `wandb.save(filename)`을 사용합니다.
2. 파일을 wandb run 디렉토리에 넣으면 run이 끝날 때 업로드됩니다.

:::info
run을 [재개](../runs/resuming.md)하고 있는 경우, `wandb.restore(filename)`을 호출하여 파일을 복구할 수 있습니다.
:::

작성 중인 파일을 동기화하려면 `wandb.save`에서 파일 이름이나 glob을 지정할 수 있습니다.

### `wandb.save`의 예시

작동 예제에 대한 [이 리포트](https://app.wandb.ai/lavanyashukla/save_and_restore/reports/Saving-and-Restoring-Models-with-W%26B--Vmlldzo3MDQ3Mw)를 참조하세요.

```python
# 현재 디렉토리에서 모델 파일 저장
wandb.save("model.h5")

# "ckpt" 문자열을 포함하는 모든 파일 저장
wandb.save("../logs/*ckpt*")

# 작성 중인 "checkpoint"로 시작하는 파일 저장
wandb.save(os.path.join(wandb.run.dir, "checkpoint*"))
```

:::info
W&B의 로컬 run 디렉토리는 기본적으로 스크립트의 `./wandb` 디렉토리 내에 있으며, 경로는 `run-20171023_105053-3o4933r0`와 같이 표시됩니다. 여기서 `20171023_105053`는 타임스탬프이고 `3o4933r0`는 run의 ID입니다. `WANDB_DIR` [환경 변수](environment-variables.md)를 설정하거나 [`wandb.init`](./launch.md)의 `dir` 키워드 인수를 절대 경로로 설정하면 파일이 해당 디렉토리 내에 쓰여집니다.
:::

### 저장 정책 및 상대 경로

`wandb.save`는 기본적으로 "**live**"로 설정된 **정책** 인수를 수락합니다. 사용 가능한 정책은 다음과 같습니다:

* **live (기본값)** - 이 파일을 wandb 서버에 즉시 동기화하고 변경 시 다시 동기화
* **now** - 이 파일을 wandb 서버에 즉시 동기화하고 변경되면 계속 동기화하지 않음
* **end** - run이 끝날 때만 파일 동기화

`wandb.save`에 **base_path** 인수도 지정할 수 있습니다. 이를 통해 디렉토리 계층 구조를 유지할 수 있습니다. 예를 들어:

```python
wandb.save(path="./results/eval/*", base_path="./results", policy="now")
```

이는 패턴과 일치하는 모든 파일이 루트가 아닌 `eval` 폴더에 저장되게 합니다.

:::info
`wandb.save`가 호출되면 제공된 경로에 있는 모든 파일을 나열하고 이를 run 디렉토리(`wandb.run.dir`)에 대한 심볼릭 링크를 만듭니다. `wandb.save`를 호출한 후 해당 경로에 새 파일을 생성하면 우리는 이를 동기화하지 않습니다. 파일을 직접 `wandb.run.dir`에 작성하거나 새 파일이 생성될 때마다 `wandb.save`를 호출해야 합니다.
:::

### wandb run 디렉토리에 파일 저장 예시

파일 `model.h5`는 `wandb.run.dir`에 저장되고 트레이닝 종료 시 업로드됩니다.

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

여기 공개 예제 페이지가 있습니다. 파일 탭에서 `model-best.h5`가 자동으로 저장된 것을 확인할 수 있습니다. 이는 기본적으로 Keras 통합에 의해 저장되며, 수동으로 체크포인트를 저장하면 run과 연관하여 저장됩니다.

[실시간 예제 보기 →](https://app.wandb.ai/wandb/neurips-demo/runs/206aacqo/files)

![](/images/experiments/example_saving_file_to_directory.png)

## 파일 복원

`wandb.restore(filename)`을 호출하면 파일을 로컬 run 디렉토리에 복원합니다. 일반적으로 `filename`은 이전 실험 run에 의해 생성되고 `wandb.save`로 클라우드에 업로드된 파일을 참조합니다. 이 호출은 파일의 로컬 복사본을 만들고 읽기 위해 열린 로컬 파일 스트림을 반환합니다.

일반적인 유스 케이스:

* 과거 run에서 생성된 모델 아키텍처나 가중치 복원(보다 복잡한 버전 관리 유스 케이스는 [Artifacts](../artifacts/intro.md)를 참조하세요).
* 실패 시 마지막 체크포인트에서 트레이닝 재개([재개](../runs/resuming.md) 섹션에서 중요한 세부 정보 참조)

### `wandb.restore`의 예시

작동 예제에 대한 [이 리포트](https://app.wandb.ai/lavanyashukla/save_and_restore/reports/Saving-and-Restoring-Models-with-W%26B--Vmlldzo3MDQ3Mw)를 참조하세요.

```python
# 특정 run에서 사용자 "vanpelt"의 "my-project"에 있는 모델 파일 복원
best_model = wandb.restore("model-best.h5", run_path="vanpelt/my-project/a1b2c3d")

# 체크포인트에서 가중치 파일 복원
# (참고: run_path가 제공되지 않으면 재개가 구성되어야 함)
weights_file = wandb.restore("weights.h5")
# 프레임워크가 파일 이름을 예상하는 경우, e.g. Keras에서
# 반환된 객체의 "name" 속성을 사용
my_predefined_model.load_weights(weights_file.name)
```

> `run_path`를 지정하지 않으면, run에 대한 [재개](../runs/resuming.md)를 구성해야 합니다. 트레이닝 외부에서 프로그램적으로 파일에 엑세스하려면 [Run API](../../ref/python/run.md)를 사용하세요.

## 자주 묻는 질문

### 파일을 무시하려면 어떻게 해야 합니까?

`wandb/settings` 파일을 편집하고 `ignore_globs`를 [globs](https://en.wikipedia.org/wiki/Glob_\(programming\))의 콤마로 구분된 리스트로 설정할 수 있습니다. 또한 `WANDB_IGNORE_GLOBS` [환경 변수](./environment-variables.md)를 설정할 수도 있습니다. 일반적인 유스 케이스는 우리가 자동으로 생성하는 git 패치가 업로드되지 않도록 하는 것입니다. 예: `WANDB_IGNORE_GLOBS=*.patch`.

### 파일 저장 디렉토리 변경

AWS S3 또는 Google Cloud Storage에 기본적으로 파일을 저장하는 경우 `events.out.tfevents.1581193870.gpt-tpu-finetune-8jzqk-2033426287는 클라우드 스토리지 URL이므로 wandb에 파일을 저장할 수 없습니다.`라는 오류가 발생할 수 있습니다.

TensorBoard 이벤트 파일 또는 우리가 동기화하고자 하는 다른 파일의 로그 디렉토리를 변경하려면, 파일을 `wandb.run.dir`에 저장하여 클라우드에 동기화되도록 하세요.

### run의 이름을 어떻게 얻나요?

스크립트 내에서 run 이름을 사용하려면 `wandb.run.name`을 사용할 수 있으며, 예를 들어 "blissful-waterfall-2"와 같은 run 이름을 얻을 수 있습니다.

디스플레이 이름에 엑세스하기 전에 run을 저장해야 합니다:

```
run = wandb.init(...)
run.save()
print(run.name)
```

### 로컬에서 저장된 모든 파일을 어떻게 푸시할 수 있나요?

스크립트의 상단에서 `wandb.init` 후 `wandb.save("*.pt")`을 한 번 호출하면, 그 패턴과 일치하는 모든 파일이 `wandb.run.dir`에 작성되면 즉시 저장됩니다.

### 클라우드 스토리지에 이미 동기화된 로컬 파일을 삭제할 수 있습니까?

이미 클라우드 스토리지에 동기화된 로컬 파일을 삭제하려면 `wandb sync --clean` 명령을 실행할 수 있습니다. 사용법에 대한 더 많은 정보는 `wandb sync --help`에서 찾을 수 있습니다.

### 코드의 상태를 복원하고 싶다면 어떻게 해야 하나요?

주어진 run을 실행할 때 코드의 상태로 돌아가려면 [커맨드라인 툴](../../ref/cli/README.md)의 `restore` 명령을 사용하세요.

```shell
# 분기를 생성하고 코드 상태를 복원합니다.
# $RUN_ID가 실행될 때의 상태로 복원
wandb restore $RUN_ID
```

### `wandb`는 코드 상태를 어떻게 캡처하나요?

스크립트에서 `wandb.init`가 호출되면, 코드가 git 리포지토리에 있는 경우 마지막 git 커밋에 대한 링크가 저장됩니다. 커밋되지 않은 변경사항이나 원격과 동기화되지 않은 변경사항이 있을 경우를 대비하여 diff 패치도 생성됩니다.