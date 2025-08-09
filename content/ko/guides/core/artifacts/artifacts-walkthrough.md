---
title: '튜토리얼: Datasets 아티팩트 생성, 추적, 그리고 사용하기'
description: Artifacts 퀵스타트에서는 W&B와 함께 Datasets 아티팩트를 생성하고, 추적하며, 사용하는 방법을 안내합니다.
displayed_sidebar: default
menu:
  default:
    identifier: ko-guides-core-artifacts-artifacts-walkthrough
---

이 안내서에서는 [W&B Runs]({{< relref path="/guides/models/track/runs/" lang="ko" >}})에서 Dataset artifact 를 생성, 추적, 활용하는 방법을 안내합니다.

## 1. W&B에 로그인하기

W&B 라이브러리를 임포트하고 W&B에 로그인하세요. 아직 계정이 없다면 무료 W&B 계정을 만들어야 합니다.

```python
import wandb

wandb.login()
```

## 2. Run 초기화

[`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ko" >}}) API를 사용하여 W&B Run 을 시작하고 데이터 로그 및 동기화의 백그라운드 프로세스를 만듭니다. 프로젝트 이름과 작업 유형(job_type)을 지정해 주세요.

```python
# W&B Run 생성 예시. Dataset artifact 를 만들 예정이므로 job_type을 'dataset'으로 설정합니다.
run = wandb.init(project="artifacts-example", job_type="upload-dataset")
```

## 3. Artifact 오브젝트 생성

[`wandb.Artifact()`]({{< relref path="/ref/python/sdk/classes/artifact.md" lang="ko" >}}) API로 artifact 오브젝트를 만듭니다. `name`은 artifact의 이름, `type`은 파일 종류(설명)를 입력합니다.

예를 들어 아래 코드는 `bicycle-dataset`이라는 이름과 `dataset` type으로 artifact 를 생성합니다:

```python
artifact = wandb.Artifact(name="bicycle-dataset", type="dataset")
```

artifact 구성 방법에 대한 자세한 내용은 [Construct artifacts]({{< relref path="./construct-an-artifact.md" lang="ko" >}}) 문서를 참고하세요.

## Dataset을 Artifact에 추가하기

artifact에 파일을 추가할 수 있습니다. 모델과 Dataset 파일이 대표적입니다. 아래는 로컬에 있는 `dataset.h5` 파일을 artifact에 담는 예시입니다:

```python
# artifact에 파일 추가
artifact.add_file(local_path="dataset.h5")
```

위 예시에서 `dataset.h5`는 실제 추가하려는 파일의 경로/이름으로 변경해 주세요.

## 4. Dataset 로그하기

W&B run 오브젝트의 `log_artifact()` 메서드를 사용해 artifact 버전을 저장하고, 해당 run의 출력(output) artifact로 선언하세요.

```python
# artifact 버전을 W&B에 저장,
# 그리고 이 run의 output으로 기록
run.log_artifact(artifact)
```

artifact를 로그하면 기본적으로 `'latest'` 에일리어스가 붙습니다. artifact의 에일리어스 및 버전에 대한 더 자세한 정보는 [Create a custom alias]({{< relref path="./create-a-custom-alias.md" lang="ko" >}}) 및 [Create new artifact versions]({{< relref path="./create-a-new-artifact-version.md" lang="ko" >}}) 문서를 참고하세요.

## 5. Artifact 다운로드 및 활용

다음 코드는 로그를 통해 W&B 서버에 저장된 artifact를 활용하는 기본적인 절차입니다.

1. 먼저, **`wandb.init()`** 으로 새 run 오브젝트를 초기화합니다.
2. 그 다음, run 오브젝트의 [`use_artifact()`]({{< relref path="/ref/python/sdk/classes/run.md#use_artifact" lang="ko" >}}) 메서드로 사용할 artifact를 지정하고, 이 artifact 오브젝트를 반환받습니다.
3. 마지막으로, artifact 오브젝트의 [`download()`]({{< relref path="/ref/python/sdk/classes/artifact.md#download" lang="ko" >}}) 메서드로 artifact 파일을 다운로드합니다.

```python
# W&B Run 생성. 이번엔 'training'을 type으로 지정한 예시
# 트레이닝 과정을 추적하기 때문입니다.
run = wandb.init(project="artifacts-example", job_type="training")

# W&B에서 artifact를 불러오고, 이 run의 입력(input)으로 명시
artifact = run.use_artifact("bicycle-dataset:latest")

# artifact 데이터 다운로드
artifact_dir = artifact.download()
```

그리고 Public API(`wandb.Api`)를 활용하면 Run과 무관하게 이미 저장된 데이터의 내보내기(또는 업데이트)도 가능합니다. 자세한 사용법은 [Track external files]({{< relref path="./track-external-files.md" lang="ko" >}}) 문서를 참고하세요.