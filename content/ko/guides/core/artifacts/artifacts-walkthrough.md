---
title: 'Tutorial: Create, track, and use a dataset artifact'
description: Artifacts 퀵스타트는 W&B로 데이터셋 아티팩트를 생성, 추적 및 사용하는 방법을 보여줍니다.
displayed_sidebar: default
menu:
  default:
    identifier: ko-guides-core-artifacts-artifacts-walkthrough
---

이 연습에서는 [W&B Runs]({{< relref path="/guides/models/track/runs/" lang="ko" >}})에서 데이터셋 아티팩트를 생성, 추적 및 사용하는 방법을 보여줍니다.

## 1. W&B에 로그인

W&B 라이브러리를 가져오고 W&B에 로그인합니다. 아직 W&B 계정이 없는 경우 무료 W&B 계정에 가입해야 합니다.

```python
import wandb

wandb.login()
```

## 2. run 초기화

[`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ko" >}}) API를 사용하여 W&B Run으로 데이터를 동기화하고 기록하는 백그라운드 프로세스를 생성합니다. 프로젝트 이름과 job 유형을 제공합니다.

```python
# W&B Run을 생성합니다. 이 예제에서는 데이터셋 아티팩트를 만드는 방법을 보여주므로 'dataset'을 job 유형으로 지정합니다.
run = wandb.init(project="artifacts-example", job_type="upload-dataset")
```

## 3. 아티팩트 오브젝트 생성

[`wandb.Artifact()`]({{< relref path="/ref/python/artifact.md" lang="ko" >}}) API로 아티팩트 오브젝트를 생성합니다. 아티팩트 이름과 파일 형식에 대한 설명을 각각 `name` 및 `type` 파라미터에 제공합니다.

예를 들어 다음 코드 조각은 `‘bicycle-dataset’`이라는 아티팩트를 `‘dataset’` 레이블로 만드는 방법을 보여줍니다.

```python
artifact = wandb.Artifact(name="bicycle-dataset", type="dataset")
```

아티팩트 구성 방법에 대한 자세한 내용은 [아티팩트 구성]({{< relref path="./construct-an-artifact.md" lang="ko" >}})을 참조하세요.

## 아티팩트에 데이터셋 추가

아티팩트에 파일을 추가합니다. 일반적인 파일 유형에는 Models 및 Datasets이 있습니다. 다음 예제에서는 머신에 로컬로 저장된 `dataset.h5`라는 데이터셋을 아티팩트에 추가합니다.

```python
# 아티팩트의 내용에 파일 추가
artifact.add_file(local_path="dataset.h5")
```

앞의 코드 조각에서 파일 이름 `dataset.h5`를 아티팩트에 추가하려는 파일의 경로로 바꿉니다.

## 4. 데이터셋 기록

W&B run 오브젝트의 `log_artifact()` 메소드를 사용하여 아티팩트 버전을 저장하고 아티팩트를 run의 출력으로 선언합니다.

```python
# 아티팩트 버전을 W&B에 저장하고 이 run의 출력으로 표시합니다.
run.log_artifact(artifact)
```

아티팩트를 기록할 때 기본적으로 `'latest'` 에일리어스가 생성됩니다. 아티팩트 에일리어스 및 버전에 대한 자세한 내용은 [커스텀 에일리어스 생성]({{< relref path="./create-a-custom-alias.md" lang="ko" >}}) 및 [새 아티팩트 버전 생성]({{< relref path="./create-a-new-artifact-version.md" lang="ko" >}})을 참조하세요.

## 5. 아티팩트 다운로드 및 사용

다음 코드 예제에서는 W&B 서버에 기록하고 저장한 아티팩트를 사용하는 단계를 보여줍니다.

1. 먼저 **`wandb.init()`**으로 새 run 오브젝트를 초기화합니다.
2. 둘째, run 오브젝트의 [`use_artifact()`]({{< relref path="/ref/python/run.md#use_artifact" lang="ko" >}}) 메소드를 사용하여 사용할 아티팩트를 W&B에 알립니다. 그러면 아티팩트 오브젝트가 반환됩니다.
3. 셋째, 아티팩트의 [`download()`]({{< relref path="/ref/python/artifact.md#download" lang="ko" >}}) 메소드를 사용하여 아티팩트의 내용을 다운로드합니다.

```python
# W&B Run을 생성합니다. 여기서는 'training'을 'type'으로 지정합니다.
# 이 run을 사용하여 트레이닝을 추적하기 때문입니다.
run = wandb.init(project="artifacts-example", job_type="training")

# W&B에서 아티팩트를 쿼리하고 이 run의 입력으로 표시합니다.
artifact = run.use_artifact("bicycle-dataset:latest")

# 아티팩트의 내용 다운로드
artifact_dir = artifact.download()
```

또는 Public API(`wandb.Api`)를 사용하여 Run 외부의 W&B에 이미 저장된 데이터를 내보내거나(또는 업데이트)할 수 있습니다. 자세한 내용은 [외부 파일 추적]({{< relref path="./track-external-files.md" lang="ko" >}})을 참조하세요.
