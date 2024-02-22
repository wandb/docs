---
description: Overview of what W&B Artifacts are, how they work, and how to get started
  using W&B Artifacts.
slug: /guides/artifacts
displayed_sidebar: default
---
import Translate, {translate} from '@docusaurus/Translate';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# 아티팩트

<W&B 아티팩트를 사용하여 직렬화된 데이터를 [W&B 실행](../runs/intro.md)의 입력 및 출력으로 추적하고 버전을 관리하세요. 예를 들어, 모델 학습 실행은 입력으로 데이터세트를 사용하고 출력으로 학습된 모델을 생성할 수 있습니다. 실행에 하이퍼파라미터와 메타데이터를 기록하는 것 외에도, 모델을 학습시키는 데 사용된 데이터세트를 입력으로 로그하고 결과 모델 체크포인트를 출력으로 로그하는 데 아티팩트를 사용할 수 있습니다. "이 모델이 어떤 버전의 데이터세트로 학습되었는지" 항상 답할 수 있습니다.

요약하자면, W&B 아티팩트를 사용하면 다음을 할 수 있습니다:
* [모델의 출처, 포함된 데이터를 훈련시킨 데이터를 확인](./explore-and-traverse-an-artifact-graph.md).
* [모든 데이터세트 변경 또는 모델 체크포인트 버전 관리](./create-a-new-artifact-version.md).
* [모델과 데이터세트를 팀 내에서 쉽게 재사용](./download-and-use-an-artifact.md).

![](/images/artifacts/artifacts_landing_page2.png)

위 다이어그램은 [실행](../runs/intro.md)의 입력 및 출력으로서 아티팩트를 사용하여 전체 ML 워크플로를 어떻게 활용할 수 있는지 보여줍니다.

## 작동 방식

네 줄의 코드로 아티팩트를 생성하세요:
1. [W&B 실행](../runs/intro.md)을 생성합니다.
2. [`wandb.Artifact`](../../ref/python/artifact.md) API로 아티팩트 개체를 생성합니다.
3. 모델 파일이나 데이터세트와 같은 하나 이상의 파일을 아티팩트 개체에 추가합니다.
4. W&B에 아티팩트를 로그합니다.

```python showLineNumbers
run = wandb.init(project="artifacts-example", job_type="add-dataset")
artifact = wandb.Artifact(name="my_data", type="dataset")
artifact.add_dir(local_path="./dataset.h5")  # 데이터세트 디렉터리를 아티팩트에 추가
run.log_artifact(artifact)  # 아티팩트 버전 "my_data:v0"을 로그함
```

:::tip
위의 코드 조각과 이 페이지에 링크된 [colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Artifacts_Quickstart_with_W&B.ipynb)은 파일을 W&B에 업로드하여 추적하는 방법을 보여줍니다. 외부 객체 저장소(예: Amazon S3 버킷)에 저장된 파일이나 디렉터리에 대한 참조를 추가하는 방법에 대한 정보는 [외부 파일 추적](./track-external-files.md) 페이지를 참조하세요.
:::

## 시작 방법

사용 사례에 따라 W&B 아티팩트를 시작하는 데 다음 리소스를 탐색하세요:

* W&B 아티팩트를 처음 사용하는 경우, [아티팩트 Colab 노트북](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Artifacts_Quickstart_with_W%B.ipynb#scrollTo=fti9TCdjOfHT)을 통해 시작하는 것이 좋습니다.
* 데이터세트 아티팩트를 생성, 추적 및 사용하기 위해 사용할 수 있는 W&B Python SDK 명령의 단계별 개요를 제공하는 [아티팩트 가이드](./artifacts-walkthrough.md)를 읽어보세요.
* 다음을 배우기 위해 이 장을 탐색하세요:
  * [아티팩트 구성](./construct-an-artifact.md) 또는 [새 아티팩트 버전 생성](./create-a-new-artifact-version.md)
  * [아티팩트 업데이트](./update-an-artifact.md)
  * [아티팩트 다운로드 및 사용](./download-and-use-an-artifact.md).
  * [아티팩트 삭제](./delete-artifacts.md).
* [Python SDK 아티팩트 API](../../ref/python/artifact.md) 및 [아티팩트 CLI 참조 가이드](../../ref/cli/wandb-artifact/README.md)를 탐색하세요.