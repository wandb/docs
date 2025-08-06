---
title: OpenAI 파인튜닝
description: W&B를 사용하여 OpenAI 모델을 파인튜닝하는 방법
menu:
  default:
    identifier: ko-guides-integrations-openai-fine-tuning
    parent: integrations
weight: 250
---

{{< cta-button colabLink="https://wandb.me/openai-colab" >}}

OpenAI GPT-3.5 또는 GPT-4 모델의 파인튜닝 메트릭과 설정을 W&B에 기록하세요. W&B 에코시스템을 활용해 파인튜닝 Experiments, Models, Datasets 를 추적하고, 동료들과 결과를 공유할 수 있습니다.

{{% alert %}}
파인튜닝할 수 있는 모델 목록은 [OpenAI 문서](https://platform.openai.com/docs/guides/fine-tuning/which-models-can-be-fine-tuned)에서 확인하세요.
{{% /alert %}}

OpenAI와 W&B를 연동해 파인튜닝하는 방법에 대한 추가 정보는 OpenAI 문서의 [W&B Integration](https://platform.openai.com/docs/guides/fine-tuning/weights-and-biases-integration) 섹션을 참고하세요.


## OpenAI Python API 설치 또는 업데이트

W&B OpenAI 파인튜닝 연동은 OpenAI 버전 1.0 이상에서 동작합니다. 최신 버전의 [OpenAI Python API](https://pypi.org/project/openai/) 라이브러리는 PyPI 문서를 참고하세요.

OpenAI Python API를 설치하려면 다음 명령어를 실행하세요:
```python
pip install openai
```

이미 OpenAI Python API가 설치되어 있다면 업데이트는 다음과 같이 할 수 있습니다:
```python
pip install -U openai
```


## OpenAI 파인튜닝 결과 동기화(Sync)

W&B를 OpenAI의 파인튜닝 API와 연동하면 파인튜닝 메트릭과 설정을 W&B로 바로 기록할 수 있습니다. 이를 위해 `wandb.integration.openai.fine_tuning` 모듈의 `WandbLogger` 클래스를 사용하면 됩니다.

```python
from wandb.integration.openai.fine_tuning import WandbLogger

# 파인튜닝 로직

WandbLogger.sync(fine_tune_job_id=FINETUNE_JOB_ID)
```

{{< img src="/images/integrations/open_ai_auto_scan.png" alt="OpenAI auto-scan feature" >}}

### 파인튜닝 결과 동기화

스크립트에서 결과를 동기화하세요.

```python
from wandb.integration.openai.fine_tuning import WandbLogger

# 한 줄 명령어
WandbLogger.sync()

# 선택적 파라미터 전달
WandbLogger.sync(
    fine_tune_job_id=None,
    num_fine_tunes=None,
    project="OpenAI-Fine-Tune",
    entity=None,
    overwrite=False,
    model_artifact_name="model-metadata",
    model_artifact_type="model",
    **kwargs_wandb_init
)
```

### 참조 (Reference)

| 인수 (Argument)              | 설명                                                                                                               |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| fine_tune_job_id         | OpenAI에서 파인튜닝 job을 생성할 때 받는 Fine-Tune ID입니다. 이 인수가 None(기본값)이면, 아직 동기화되지 않은 모든 OpenAI 파인튜닝 job이 W&B로 동기화됩니다. |
| openai_client            | 초기화된 OpenAI client를 `sync`에 전달할 수 있습니다. 제공하지 않으면 Logger가 자체적으로 초기화합니다. 기본값은 None입니다. |
| num_fine_tunes           | ID를 제공하지 않으면, 아직 동기화되지 않은 모든 파인튜닝 작업이 W&B에 기록됩니다. 이 인수로 최근 파인튜닝 결과 n개만 동기화하도록 선택할 수 있습니다. 예: num_fine_tunes가 5면 최근 5개의 파인튜닝만 선택됩니다. |
| project                  | 파인튜닝 메트릭, 모델, 데이터 등이 기록될 W&B 프로젝트 이름입니다. 기본 프로젝트명은 "OpenAI-Fine-Tune" 입니다. |
| entity                   | Results를 전송할 W&B 사용자명 또는 팀명입니다. 기본적으로 내 기본 entity(대부분 사용자명)가 사용됩니다. |
| overwrite                | 동일한 파인튜닝 job의 기존 wandb run을 강제로 덮어쓰면서 기록합니다. 기본값은 False입니다. |
| wait_for_job_success     | OpenAI 파인튜닝 job이 시작되면 약간의 시간이 소요됩니다. 파인튜닝이 끝나면 메트릭이 W&B에 바로 기록되도록, 이 설정은 60초마다 파인튜닝 job의 상태가 `succeeded`로 바뀌었는지 확인합니다. 성공적으로 감지되면 메트릭이 자동 동기화됩니다. 기본값은 True입니다. |
| model_artifact_name      | 기록되는 모델 artifact의 이름입니다. 기본값은 `"model-metadata"`입니다. |
| model_artifact_type      | 기록되는 모델 artifact의 타입입니다. 기본값은 `"model"`입니다. |
| \*\*kwargs_wandb_init  | [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ko" >}})에 직접 전달할 추가 인수입니다. |

## Dataset 버전 관리 및 시각화

### 버전 관리

OpenAI에 파인튜닝을 위해 업로드한 트레이닝/검증 데이터는 자동으로 W&B Artifacts로 기록되어 손쉬운 버전 관리를 할 수 있습니다. 아래는 Artifacts 내 트레이닝 파일 예시입니다. 이 파일을 기록한 W&B run, 기록 시점, 데이터셋 버전, 메타데이터, 그리고 트레이닝 데이터에서 학습된 모델로의 DAG 계보까지 확인할 수 있습니다.

{{< img src="/images/integrations/openai_data_artifacts.png" alt="W&B Artifacts with training datasets" >}}

### 시각화

데이터셋은 W&B Tables로 시각화되어, 데이터셋을 탐색, 검색, 상호작용할 수 있습니다. 아래는 W&B Tables를 활용해 시각화한 트레이닝 샘플입니다.

{{< img src="/images/integrations/openai_data_visualization.png" alt="OpenAI data" >}}


## 파인튜닝된 모델과 모델 버전 관리

OpenAI는 파인튜닝된 모델의 ID를 제공합니다. 모델 가중치에는 직접 엑세스할 수 없지만, `WandbLogger`는 모델에 대한 모든 정보(하이퍼파라미터, 데이터 파일 ID 등)와 함께 `fine_tuned_model` ID를 포함한 `model_metadata.json` 파일을 생성해 W&B Artifact로 기록합니다.

이 모델(메타데이터) Artifact는 [W&B Registry]({{< relref path="/guides/core/registry/" lang="ko" >}}) 내의 모델과 연결될 수 있습니다.

{{< img src="/images/integrations/openai_model_metadata.png" alt="OpenAI model metadata" >}}


## 자주 묻는 질문 (FAQ)

### 내 파인튜닝 결과를 W&B에서 팀과 공유하려면?

아래와 같이 `entity` 값을 팀 계정으로 지정해 로그를 남기세요:

```python
WandbLogger.sync(entity="YOUR_TEAM_NAME")
```

### Run을 어떻게 조직화할 수 있나요?

W&B의 Runs는 자동으로 정리되며, job 유형, 베이스 모델, 러닝레이트, 트레이닝 파일명, 기타 하이퍼파라미터 등 원하는 설정 파라미터 기준으로 필터링/정렬할 수 있습니다.

또한 Run 이름을 변경하거나, 노트를 추가하거나, 태그를 생성해 그룹화할 수 있습니다.

원하는 대로 정리한 후 Workspace를 저장하고, 필요한 데이터와 Artifacts(트레이닝/검증 파일 등)를 가져와 Report 생성에 활용할 수 있습니다.

### 파인튜닝한 모델에는 어떻게 엑세스하나요?

파인튜닝된 모델의 ID는 artifacts(`model_metadata.json`)와 설정으로 W&B에 기록됩니다.

```python
import wandb
    
with wandb.init(project="OpenAI-Fine-Tune", entity="YOUR_TEAM_NAME") as run:
    ft_artifact = run.use_artifact("ENTITY/PROJECT/model_metadata:VERSION")
    artifact_dir = ft_artifact.download()
```

여기서 `VERSION`은 다음 중 하나가 될 수 있습니다:

* `v2`와 같은 버전 넘버
* `ft-xxxxxxxxx` 형식의 파인튜닝 id
* `latest` 등 자동 에일리어스, 또는 수동으로 추가한 에일리어스

다운로드한 `model_metadata.json` 파일을 읽어서 `fine_tuned_model` id를 확인할 수 있습니다.

### 파인튜닝 결과가 성공적으로 동기화되지 않았다면?

파인튜닝 결과가 W&B에 제대로 기록되지 않았다면, `overwrite=True` 옵션과 함께 job id를 전달할 수 있습니다:

```python
WandbLogger.sync(
    fine_tune_job_id="FINE_TUNE_JOB_ID",
    overwrite=True,
)
```

### W&B로 내 데이터셋과 모델을 추적할 수 있나요?

트레이닝/검증 데이터는 W&B에 Artifacts로 자동 기록됩니다. 파인튜닝된 모델의 ID 등 메타데이터 역시 Artifacts로 남습니다.

`wandb.Artifact`, `wandb.Run.log` 등과 같은 wandb의 저수준 API를 사용해 파이프라인을 직접 제어하는 것도 가능합니다. 이를 통해 데이터와 모델의 추적성을 완벽히 확보할 수 있습니다.

{{< img src="/images/integrations/open_ai_faq_can_track.png" alt="OpenAI tracking FAQ" >}}

## 참고 자료

* [OpenAI 파인튜닝 공식 문서](https://platform.openai.com/docs/guides/fine-tuning/)에는 다양한 팁과 상세 정보가 정리되어 있습니다.
* [Demo Colab](https://wandb.me/openai-colab)
* [How to Fine-Tune Your OpenAI GPT-3.5 and GPT-4 Models with W&B](https://wandb.me/openai-report) 리포트