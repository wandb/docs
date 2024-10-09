---
title: Manage artifact data retention
description: TTL 정책 (TTL)
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/kas-artifacts-ttl-colab/colabs/wandb-artifacts/WandB_Artifacts_Time_to_live_TTL_Walkthrough.ipynb"/>

W&B의 Artifacts 유효기간 (TTL) 정책으로 아티팩트를 삭제하는 시점을 예약하세요. 아티팩트를 삭제하면, W&B는 해당 아티팩트를 *소프트 삭제*로 표시합니다. 즉, 아티팩트가 삭제 대상으로 표시되지만 파일은 즉시 스토리지에서 삭제되지 않습니다. W&B가 아티팩트를 삭제하는 방법에 대한 자세한 내용은 [아티팩트 삭제](./delete-artifacts.md) 페이지를 참조하세요.

W&B 앱에서 Artifacts TTL로 데이터 보존을 관리하는 방법을 배우려면 [이](https://www.youtube.com/watch?v=hQ9J6BoVmnc) 비디오 튜토리얼을 확인하세요.

:::note
모델 레지스트리에 연결된 모델 아티팩트의 TTL 정책 설정 옵션은 비활성화됩니다. 이는 프로덕션 워크플로우에 사용되는 연결된 모델이 실수로 만료되지 않도록 하기 위한 것입니다.
:::
:::info
* 팀의 설정을 보고 팀 레벨 TTL 설정(1) TTL 정책을 설정하거나 편집할 수 있는 권한 부여, (2) 팀 기본 TTL 설정)을 엑세스할 수 있는 권한은 팀 관리자에게만 있습니다.  
* W&B 앱 UI에서 아티팩트 세부정보에 TTL 정책을 설정하거나 편집할 수 있는 옵션을 찾을 수 없거나 프로그램적으로 TTL을 설정해도 아티팩트의 TTL 속성이 성공적으로 변경되지 않는 경우, 팀 관리자가 해당 권한을 부여하지 않은 것입니다.
:::

## 자동 생성 아티팩트
사용자가 생성한 아티팩트에만 TTL 정책을 사용할 수 있습니다. W&B가 자동 생성한 아티팩트에는 TTL 정책을 설정할 수 없습니다.

다음 아티팩트 유형은 자동 생성된 아티팩트를 나타냅니다:
- `run_table`
- `code`
- `job`
- `wandb-*`로 시작하는 모든 아티팩트 유형

아티팩트의 유형은 [W&B 플랫폼](../artifacts/explore-and-traverse-an-artifact-graph.md)에서 또는 프로그램적으로 확인할 수 있습니다:

```python
import wandb
run = wandb.init(project="<my-project-name>")
artifact = run.use_artifact(artifact_or_name="<my-artifact-name>")
print(artifact.type)
```

`<>`로 감싸진 값은 사용자 본인의 값으로 대체하세요.

## TTL 정책 설정 및 편집 권한 정의
팀 내에서 TTL 정책을 설정하고 편집할 수 있는 대상을 정의하세요. 팀 관리자에게만 TTL 권한을 부여하거나, 팀 관리자와 팀 멤버 모두에게 TTL 권한을 부여할 수 있습니다.

:::info
TTL 정책을 설정하거나 편집할 수 있는 대상을 정의할 수 있는 권한은 팀 관리자에게만 있습니다.
:::

1. 팀 프로필 페이지로 이동합니다.
2. **Settings** 탭을 선택합니다.
3. **Artifacts time-to-live (TTL) section**으로 이동합니다.
4. **TTL permissions dropdown**에서 TTL 정책을 설정하고 편집할 수 있는 대상을 선택합니다.  
5. **Review and save settings** 버튼을 클릭합니다.
6. 변경 내용을 확인하고 **Save settings**를 선택합니다.

![](/images/artifacts/define_who_sets_ttl.gif)

## TTL 정책 생성
아티팩트를 만들 때 또는 아티팩트를 생성한 후에 TTL 정책을 설정합니다.

아래의 모든 코드조각에서 `<>`로 감싸진 내용을 본인의 정보로 대체하여 코드조각을 사용하세요.

### 아티팩트를 생성할 때 TTL 정책 설정
W&B Python SDK를 사용하여 아티팩트를 생성할 때 TTL 정책을 정의합니다. TTL 정책은 일반적으로 일 단위로 정의됩니다.

:::tip
아티팩트를 생성할 때 TTL 정책을 정의하는 것은 일반적으로 [아티팩트를 생성하는](./construct-an-artifact.md) 방법과 유사합니다. 차이점은 아티팩트의 `ttl` 속성에 시간 델타를 전달하는 것입니다.
:::

절차는 다음과 같습니다:

1. [아티팩트를 생성합니다](./construct-an-artifact.md).
2. 파일, 디렉토리 또는 참조와 같은 내용을 [아티팩트에 추가합니다](./construct-an-artifact.md#add-files-to-an-artifact).
3. Python의 표준 라이브러리의 일부인 [`datetime.timedelta`](https://docs.python.org/3/library/datetime.html) 데이터 유형으로 TTL 시간 제한을 정의합니다.
4. [아티팩트를 로그](./construct-an-artifact.md#3-save-your-artifact-to-the-wb-server)합니다.

다음 코드조각은 아티팩트를 생성하고 TTL 정책을 설정하는 방법을 보여줍니다.

```python
import wandb
from datetime import timedelta

run = wandb.init(project="<my-project-name>", entity="<my-entity>")
artifact = wandb.Artifact(name="<artifact-name>", type="<type>")
artifact.add_file("<my_file>")

artifact.ttl = timedelta(days=30)  # TTL 정책 설정
run.log_artifact(artifact)
```

위의 코드조각은 아티팩트에 대한 TTL 정책을 30일로 설정합니다. 즉, W&B는 30일 후에 아티팩트를 삭제합니다.

### 아티팩트를 생성한 후에 TTL 정책 설정 또는 편집
이미 존재하는 아티팩트에 대한 TTL 정책을 정의하려면 W&B App UI 또는 W&B Python SDK를 사용하세요.

:::note
아티팩트의 TTL을 수정하면, 아티팩트의 만료 시간은 여전히 아티팩트의 `createdAt` 타임스탬프를 기준으로 계산됩니다.
:::

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python SDK', value: 'python'},
    {label: 'W&B App', value: 'app'},
  ]}>

  <TabItem value="python">

1. [아티팩트를 가져옵니다](./download-and-use-an-artifact.md).
2. 아티팩트의 `ttl` 속성에 시간 델타를 전달합니다.
3. [`save`](../../ref/python/run.md#save) 메소드를 사용하여 아티팩트를 업데이트합니다.

다음 코드조각은 아티팩트에 TTL 정책을 설정하는 방법을 보여줍니다:
```python
import wandb
from datetime import timedelta

artifact = run.use_artifact("<my-entity/my-project/my-artifact:alias>")
artifact.ttl = timedelta(days=365 * 2)  # 2년 후 삭제
artifact.save()
```

위의 코드 예제는 TTL 정책을 2년으로 설정합니다.

  </TabItem>
  <TabItem value="app">

1. W&B App UI에서 W&B 프로젝트로 이동합니다.
2. 왼쪽 패널에서 아티팩트 아이콘을 선택합니다.
3. 아티팩트 목록에서 아티팩트 유형을 확장합니다.
4. TTL 정책을 편집할 아티팩트 버전을 선택합니다.
5. **Version** 탭을 클릭합니다.
6. 드롭다운에서 **Edit TTL policy**를 선택합니다.
7. 나타나는 모달에서 TTL 정책 드롭다운에서 **Custom**을 선택합니다.
8. **TTL duration** 필드에서 TTL 정책을 일 단위로 설정합니다.
9. **Update TTL** 버튼을 선택하여 변경 사항을 저장합니다.

![](/images/artifacts/edit_ttl_ui.gif)

  </TabItem>
</Tabs>

### 팀에 대한 기본 TTL 정책 설정

:::info
기본 TTL 정책을 설정할 수 있는 권한은 팀 관리자에게만 있습니다.
:::

팀에 기본 TTL 정책을 설정합니다. 기본 TTL 정책은 각각의 생성일에 기반하여 모든 기존 및 미래의 아티팩트에 적용됩니다. 이미 있는 버전 레벨 TTL 정책이 있는 아티팩트는 팀의 기본 TTL에 영향을 받지 않습니다.

1. 팀 프로필 페이지로 이동합니다.
2. **Settings** 탭을 선택합니다.
3. **Artifacts time-to-live (TTL) section**으로 이동합니다.
4. **Set team's default TTL policy**를 클릭합니다.
5. **Duration** 필드에서 TTL 정책을 일 단위로 설정합니다.
6. **Review and save settings**를 클릭합니다.
7. 변경 내용을 확인한 다음 **Save settings**를 선택합니다.

![](/images/artifacts/set_default_ttl.gif)

## TTL 정책 비활성화
특정 아티팩트 버전에 대한 TTL 정책을 비활성화하려면 W&B Python SDK 또는 W&B App UI를 사용하세요.

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python SDK', value: 'python'},
    {label: 'W&B App', value: 'app'},
  ]}>

  <TabItem value="python">

1. [아티팩트를 가져옵니다](./download-and-use-an-artifact.md).
2. 아티팩트의 `ttl` 속성을 `None`으로 설정합니다.
3. [`save`](../../ref/python/run.md#save) 메소드를 사용하여 아티팩트를 업데이트합니다.

다음 코드조각은 아티팩트에 대한 TTL 정책을 비활성화하는 방법을 보여줍니다:
```python
artifact = run.use_artifact("<my-entity/my-project/my-artifact:alias>")
artifact.ttl = None
artifact.save()
```

  </TabItem>
  <TabItem value="app">

1. W&B App UI에서 W&B 프로젝트로 이동합니다.
2. 왼쪽 패널에서 아티팩트 아이콘을 선택합니다.
3. 아티팩트 목록에서 아티팩트 유형을 확장합니다.
4. TTL 정책을 편집할 아티팩트 버전을 선택합니다.
5. **Version** 탭을 클릭합니다.
6. **Link to registry** 버튼 옆의 meatball UI 아이콘을 클릭합니다.
7. 드롭다운에서 **Edit TTL policy**를 선택합니다.
8. 나타나는 모달에서 TTL 정책 드롭다운에서 **Deactivate**를 선택합니다.
9. **Update TTL** 버튼을 선택하여 변경 사항을 저장합니다.

![](/images/artifacts/remove_ttl_polilcy.gif)

  </TabItem>
</Tabs>

## TTL 정책 보기
Python SDK 또는 W&B App UI를 사용하여 아티팩트에 대한 TTL 정책을 볼 수 있습니다.

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python SDK', value: 'python'},
    {label: 'W&B App', value: 'app'},
  ]}>

  <TabItem value="python">

프린트 문을 사용하여 아티팩트의 TTL 정책을 확인하세요. 다음 예제는 아티팩트를 검색하고 TTL 정책을 보는 방법을 보여줍니다:

```python
artifact = run.use_artifact("<my-entity/my-project/my-artifact:alias>")
print(artifact.ttl)
```

  </TabItem>
  <TabItem value="app">

W&B App UI를 사용하여 아티팩트에 대한 TTL 정책을 봅니다.

1. [https://wandb.ai](https://wandb.ai)에서 W&B 앱으로 이동합니다.
2. W&B 프로젝트로 이동합니다.
3. 프로젝트 내에서 왼쪽 사이드바의 Artifacts 탭을 선택합니다.
4. 컬렉션을 클릭합니다.

선택된 컬렉션 내에서 모든 아티팩트를 볼 수 있습니다. `Time to Live` 열에서 해당 아티팩트에 할당된 TTL 정책을 볼 수 있습니다.

![](/images/artifacts/ttl_collection_panel_ui.png)

  </TabItem>
</Tabs>