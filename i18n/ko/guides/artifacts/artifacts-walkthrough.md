---
description: Artifacts quickstart shows how to create, track, and use a dataset artifact
  with W&B.
displayed_sidebar: default
---

# 연습

<head>
  <title>연습</title>
</head>

다음 연습은 데이터셋 아티팩트를 생성, 추적 및 사용하기 위해 사용되는 주요 W&B Python SDK 코맨드를 보여줍니다. [W&B Runs](../runs/intro.md)에서 어떻게 사용되는지를 보여줍니다.

## 1. W&B에 로그인하기

W&B 라이브러리를 가져오고 W&B에 로그인합니다. 만약 아직 가입하지 않았다면 무료 W&B 계정에 가입해야 합니다.

```python
import wandb

wandb.login()
```

## 2. run 초기화하기

[`wandb.init()`](../../ref/python/init.md) API를 사용하여 W&B Run으로 데이터를 동기화하고 로그하는 백그라운드 프로세스를 생성합니다. 프로젝트 이름과 job 유형을 제공합니다:

```python
# W&B Run을 생성합니다. 여기서는 이 예제가 데이터셋 아티팩트를 생성하는 방법을 보여주기 때문에
# job 유형으로 'dataset'을 지정합니다.
run = wandb.init(project="artifacts-example", job_type="upload-dataset")
```

## 3. 아티팩트 오브젝트 생성하기

[`wandb.Artifact()`](../../ref/python/artifact.md) API로 아티팩트 오브젝트를 생성합니다. `name`과 `type` 파라미터에 대해 아티팩트의 이름과 파일 타입의 설명을 제공합니다.

예를 들어, 다음 코드조각은 `‘dataset’` 라벨을 가진 `‘bicycle-dataset’`이라는 아티팩트를 생성하는 방법을 보여줍니다:

```python
artifact = wandb.Artifact(name="bicycle-dataset", type="dataset")
```

아티팩트를 구성하는 방법에 대한 자세한 정보는 [아티팩트 구성하기](./construct-an-artifact.md)를 참조하세요.

## 아티팩트에 데이터셋 추가하기

아티팩트에 파일을 추가합니다. 일반적인 파일 타입에는 모델과 데이터셋이 포함됩니다. 다음 예제는 우리 기계에 로컬로 저장된 `dataset.h5`라는 이름의 데이터셋을 아티팩트에 추가하는 방법을 보여줍니다:

```python
# 아티팩트의 내용에 파일 추가하기
artifact.add_file(local_path="dataset.h5")
```

위 코드조각에서 `dataset.h5` 파일 이름을 아티팩트에 추가하고자 하는 파일의 경로로 교체하세요.

## 4. 데이터셋 로그하기

W&B run 오브젝트의 `log_artifact()` 메소드를 사용하여 아티팩트 버전을 저장하고 run의 출력으로 아티팩트를 선언합니다.

```python
# 아티팩트 버전을 W&B에 저장하고
# 이 run의 출력으로 표시하기
run.log_artifact(artifact)
```

아티팩트를 로그할 때 기본적으로 `'latest'` 에일리어스가 생성됩니다. 아티팩트 에일리어스와 버전에 대한 자세한 정보는 각각 [사용자 정의 에일리어스 만들기](./create-a-custom-alias.md)와 [새 아티팩트 버전 생성하기](./create-a-new-artifact-version.md)를 참조하세요.

## 5. 아티팩트 다운로드 및 사용하기

다음 코드 예제는 로그하고 W&B 서버에 저장된 아티팩트를 사용하는 단계를 보여줍니다.

1. 먼저, **`wandb.init()`**으로 새로운 run 오브젝트를 초기화합니다.
2. 둘째, run 오브젝트의 [`use_artifact()`](../../ref/python/run.md#use_artifact) 메소드를 사용하여 W&B에 어떤 아티팩트를 사용할지 알립니다. 이것은 아티팩트 오브젝트를 반환합니다.
3. 셋째, 아티팩트의 [`download()`](../../ref/python/artifact.md#download) 메소드를 사용하여 아티팩트의 내용을 다운로드합니다.

```python
# W&B Run을 생성합니다. 여기서는 'type'으로 'training'을 지정합니다
# 왜냐하면 우리는 이 run을 트레이닝 추적에 사용할 것이기 때문입니다.
run = wandb.init(project="artifacts-example", job_type="training")

# W&B에서 아티팩트를 조회하고 이 run의 입력으로 표시하기
artifact = run.use_artifact("bicycle-dataset:latest")

# 아티팩트의 내용 다운로드하기
artifact_dir = artifact.download()
```

대안으로, Public API (`wandb.Api`)를 사용하여 Run 외부에 이미 저장된 데이터를 내보내거나(또는 업데이트) 데이터를 추적할 수 있습니다. 자세한 정보는 [외부 파일 추적하기](./track-external-files.md)를 참조하세요.