---
title: Tutorial: Create, track, and use a dataset artifact
description: Artifacts 퀵스타트에서는 W&B를 사용하여 데이터셋 아티팩트를 생성, 추적 및 사용하는 방법을 보여줍니다.
displayed_sidebar: default
---

이 설명서는 [W&B Runs](../runs/intro.md)에서 데이터셋 아티팩트를 생성하고, 추적하고, 사용하는 방법을 보여줍니다.

## 1. W&B에 로그인

W&B 라이브러리를 임포트하고 W&B에 로그인하세요. 아직 가입하지 않았다면 무료 W&B 계정을 가입해야 합니다.

```python
import wandb

wandb.login()
```

## 2. run 초기화

[`wandb.init()`](../../ref/python/init.md) API를 사용하여 W&B Run으로 데이터를 동기화하고 로그하기 위한 백그라운드 프로세스를 생성합니다. 프로젝트 이름과 job 유형을 제공하세요:

```python
# W&B Run을 생성합니다. 여기서는 데이터셋 아티팩트를
# 생성하는 예시이므로 'dataset'을 job 유형으로 지정합니다.
run = wandb.init(project="artifacts-example", job_type="upload-dataset")
```

## 3. 아티팩트 오브젝트 생성

[`wandb.Artifact()`](../../ref/python/artifact.md) API를 사용하여 아티팩트 오브젝트를 생성합니다. 아티팩트의 `name`과 `type` 파라미터로 파일 유형에 대한 이름과 설명을 제공하세요.

예를 들어, 다음 코드조각은 `‘bicycle-dataset’`라는 이름의 아티팩트를 `‘dataset’` 라벨로 생성하는 방법을 보여줍니다:

```python
artifact = wandb.Artifact(name="bicycle-dataset", type="dataset")
```

아티팩트 구성에 대한 자세한 내용은 [Construct artifacts](./construct-an-artifact.md)를 참조하세요.

## 아티팩트에 데이터셋 추가

파일을 아티팩트에 추가하세요. 일반적인 파일 유형으로 모델 및 데이터셋이 포함됩니다. 다음 예는 로컬 기계에 저장된 `dataset.h5`라는 데이터셋을 아티팩트에 추가합니다:

```python
# 아티팩트의 콘텐츠에 파일 추가
artifact.add_file(local_path="dataset.h5")
```

이전 코드조각의 파일명 `dataset.h5`을 추가하려는 파일의 경로로 바꿔주세요.

## 4. 데이터셋 로그

W&B run 오브젝트의 `log_artifact()` 메소드를 사용하여 아티팩트 버전을 저장하고, 이를 run의 출력으로 선언하세요.

```python
# W&B에 아티팩트 버전을 저장하고
# 이를 이 run의 출력으로 표시합니다.
run.log_artifact(artifact)
```

아티팩트를 로그할 때 기본적으로 `'latest'` 에일리어스가 생성됩니다. 아티팩트 에일리어스와 버전에 대한 자세한 내용은 [Create a custom alias](./create-a-custom-alias.md)와 [Create new artifact versions](./create-a-new-artifact-version.md)를 각각 참조하세요.

## 5. 아티팩트 다운로드 및 사용

다음 코드 예제는 로그 및 저장한 아티팩트를 W&B 서버에서 사용하는 단계들을 보여줍니다.

1. 먼저, **`wandb.init()`** 으로 새로운 run 오브젝트를 초기화합니다.
2. 두 번째로, run 오브젝트의 [`use_artifact()`](../../ref/python/run.md#use_artifact) 메소드를 사용하여 W&B에 사용할 아티팩트를 지정합니다. 이 메소드는 아티팩트 오브젝트를 반환합니다.
3. 세 번째로, 아티팩트의 [`download()`](../../ref/python/artifact.md#download) 메소드를 사용하여 아티팩트의 내용을 다운로드합니다.

```python
# W&B Run 생성. 여기서는 'type'에 'training'을 지정합니다.
# 이 run을 트레이닝을 추적하기 위해 사용할 것입니다.
run = wandb.init(project="artifacts-example", job_type="training")

# W&B에서 아티팩트를 쿼리하고 이를 이 run의 입력으로 표시합니다.
artifact = run.use_artifact("bicycle-dataset:latest")

# 아티팩트의 내용을 다운로드합니다.
artifact_dir = artifact.download()
```

또한, Public API (`wandb.Api`)를 사용하여 Run 외부의 W&B에 이미 저장된 데이터를 내보내거나(또는 업데이트) 할 수 있습니다. 자세한 내용은 [Track external files](./track-external-files.md)를 참조하세요.