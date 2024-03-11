---
description: Overview of what W&B Artifacts are, how they work, and how to get started
  using W&B Artifacts.
slug: /guides/artifacts
displayed_sidebar: default
---
import Translate, {translate} from '@docusaurus/Translate';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# Artifacts

<CTAButtons productLink="https://wandb.ai/wandb/arttest/artifacts/model/iv3_trained/5334ab69740f9dda4fed/lineage" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Pipeline_Versioning_with_W%26B_Artifacts.ipynb"/>

W&B Artifacts를 사용하여 시리얼화된 데이터를 [W&B Runs](../runs/intro.md)의 입력 및 출력으로 추적하고 버전 관리할 수 있습니다. 예를 들어, 모델 트레이닝 run은 입력으로 데이터셋을, 출력으로 트레인된 모델을 사용할 수 있습니다. run에 하이퍼파라미터와 메타데이터를 로깅하는 것 외에도, 모델을 트레이닝하는 데 사용된 데이터셋을 입력으로, 결과 모델 체크포인트를 출력으로 로깅하기 위해 아티팩트를 사용할 수 있습니다. "이 모델이 어떤 버전의 데이터셋으로 트레이닝되었는가"라는 질문에 항상 답할 수 있습니다.

요약하자면, W&B Artifacts를 사용하면 다음을 할 수 있습니다:
* [모델의 출처, 포함한 트레이닝된 데이터를 확인](./explore-and-traverse-an-artifact-graph.md).
* [모든 데이터셋 변경이나 모델 체크포인트 버전 관리](./create-a-new-artifact-version.md).
* [팀 전체에서 모델과 데이터셋을 쉽게 재사용](./download-and-use-an-artifact.md).

![](/images/artifacts/artifacts_landing_page2.png)

위 다이어그램은 [runs](../runs/intro.md)의 입력 및 출력으로 아티팩트를 사용하여 전체 ML 워크플로우를 어떻게 활용할 수 있는지 보여줍니다.

## 작동 방식

네 줄의 코드로 아티팩트를 생성합니다:
1. [W&B run](../runs/intro.md)을 생성합니다.
2. [`wandb.Artifact`](../../ref/python/artifact.md) API를 사용하여 아티팩트 오브젝트를 생성합니다.
3. 하나 이상의 파일(예: 모델 파일 또는 데이터셋)을 아티팩트 오브젝트에 추가합니다.
4. W&B에 아티팩트를 로그합니다.


```python showLineNumbers
run = wandb.init(project="artifacts-example", job_type="add-dataset")
artifact = wandb.Artifact(name="my_data", type="dataset")
artifact.add_dir(local_path="./dataset.h5")  # 아티팩트에 데이터셋 디렉토리 추가
run.log_artifact(artifact)  # "my_data:v0"라는 아티팩트 버전을 로그
```

:::tip
앞선 코드조각과 이 페이지에 링크된 [colab 노트북](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Artifacts_Quickstart_with_W&B.ipynb)은 파일을 W&B에 업로드하여 추적하는 방법을 보여줍니다. 외부 오브젝트 스토리지(예: Amazon S3 버킷)에 저장된 파일이나 디렉토리에 대한 참조를 추가하는 방법에 대해서는 [외부 파일 추적](./track-external-files.md) 페이지를 참조하세요.
:::

## 시작 방법

W&B Artifacts를 시작하는 데 따라 다음 자료를 탐색하세요:

* W&B Artifacts를 처음 사용하는 경우 [Artifacts Colab 노트북](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Artifacts_Quickstart_with_W&B.ipynb#scrollTo=fti9TCdjOfHT)을 살펴보는 것이 좋습니다.
* 데이터셋 아티팩트를 생성, 추적 및 사용하기 위해 사용할 수 있는 W&B Python SDK 코맨드의 단계별 개요를 확인하려면 [아티팩트 가이드](./artifacts-walkthrough.md)를 읽어보세요.
* 이 챕터를 탐색하여 다음을 학습하세요:
  * [아티팩트 구성](./construct-an-artifact.md) 또는 [새로운 아티팩트 버전 생성](./create-a-new-artifact-version.md)
  * [아티팩트 업데이트](./update-an-artifact.md)
  * [아티팩트 다운로드 및 사용](./download-and-use-an-artifact.md).
  * [아티팩트 삭제](./delete-artifacts.md).
* [Python SDK Artifact APIs](../../ref/python/artifact.md) 및 [Artifact CLI 참조 가이드](../../ref/cli/wandb-artifact/README.md)를 탐색하세요.