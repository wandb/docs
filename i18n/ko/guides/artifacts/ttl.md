---
description: Time to live policies (TTL)
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# 아티팩트 TTL 정책으로 데이터 보존 관리하기

W&B 아티팩트 수명 시간(TTL) 정책으로 W&B에서 아티팩트가 삭제되는 시기를 예약하세요. 아티팩트를 삭제할 때, W&B는 해당 아티팩트를 *소프트 삭제*로 표시합니다. 즉, 아티팩트는 삭제 대상으로 표시되지만 파일은 즉시 저장소에서 삭제되지 않습니다. W&B에서 아티팩트를 삭제하는 방법에 대한 자세한 정보는 [아티팩트 삭제](./delete-artifacts.md) 페이지를 참조하세요.

W&B 앱에서 아티팩트 TTL을 사용하여 데이터 보존을 관리하는 방법을 알아보려면 [이](https://www.youtube.com/watch?v=hQ9J6BoVmnc) 비디오 튜토리얼을 확인하세요.

:::note
W&B는 모델 레지스트리에 연결된 모델 아티팩트에 대해 TTL 정책을 설정하는 옵션을 비활성화합니다. 이는 연결된 모델이 프로덕션 워크플로에서 실수로 만료되지 않도록 하기 위함입니다.
:::
:::info
* 팀 관리자만 [팀의 설정](../app/settings-page/team-settings.md)을 볼 수 있으며 팀 수준의 TTL 설정에 엑세스할 수 있습니다. 예를 들어, (1) 누가 TTL 정책을 설정하거나 편집할 수 있는지 허용하거나 (2) 팀 기본 TTL을 설정합니다.
* W&B 앱 UI에서 아티팩트의 세부 정보에서 TTL 정책을 설정하거나 편집하는 옵션을 보지 못하거나 프로그래밍 방식으로 TTL을 설정해도 아티팩트의 TTL 속성이 성공적으로 변경되지 않는 경우, 팀 관리자가 권한을 부여하지 않은 것입니다.
:::

## TTL 정책을 편집하고 설정할 수 있는 사람 정의하기
팀 내에서 누가 TTL 정책을 설정하고 편집할 수 있는지 정의하세요. 팀 관리자에게만 TTL 권한을 부여하거나 팀 관리자와 팀 멤버 모두에게 TTL 권한을 부여할 수 있습니다.

:::info
팀 관리자만 누가 TTL 정책을 설정하거나 편집할 수 있는지 정의할 수 있습니다.
:::

1. 팀의 프로필 페이지로 이동합니다.
2. **설정** 탭을 선택합니다.
3. **아티팩트 수명 시간 (TTL) 섹션**으로 이동합니다.
4. **TTL 권한 드롭다운**에서 누가 TTL 정책을 설정하고 편집할 수 있는지 선택합니다.
5. **설정 검토 및 저장**을 클릭합니다.
6. 변경 사항을 확인하고 **설정 저장**을 선택합니다.

![](/images/artifacts/define_who_sets_ttl.gif)

## TTL 정책 생성하기
아티팩트를 생성할 때 또는 아티팩트가 생성된 후 소급적으로 TTL 정책을 설정하세요.

아래 코드 조각에서 `<>`로 묶인 내용을 귀하의 정보로 교체하여 코드 조각을 사용합니다.

### 아티팩트를 생성할 때 TTL 정책 설정하기
W&B Python SDK를 사용하여 아티팩트를 생성할 때 TTL 정책을 정의합니다. TTL 정책은 일반적으로 일 단위로 정의됩니다.

:::tip
아티팩트를 생성할 때 TTL 정책을 정의하는 것은 [아티팩트 생성](./construct-an-artifact.md)하는 방법과 유사합니다. 단, 아티팩트의 `ttl` 속성에 시간 차이를 전달한다는 점이 다릅니다.
:::

단계는 다음과 같습니다:

1. [아티팩트 생성](./construct-an-artifact.md).
2. 파일, 디렉터리 또는 참조와 같은 [아티팩트에 내용 추가](./construct-an-artifact.md#add-files-to-an-artifact).
3. Python 표준 라이브러리의 일부인 [`datetime.timedelta`](https://docs.python.org/3/library/datetime.html) 데이터 타입을 사용하여 TTL 시간 제한을 정의합니다.
4. [W&B 서버에 아티팩트 로그](./construct-an-artifact.md#3-save-your-artifact-to-the-wb-server).

다음 코드 조각은 아티팩트를 생성하고 TTL 정책을 설정하는 방법을 보여줍니다.

```python
import wandb
from datetime import timedelta

run = wandb.init(project="<my-project-name>", entity="<my-entity>")
artifact = wandb.Artifact(name="<artifact-name>", type="<type>")
artifact.add_file("<my_file>")

artifact.ttl = timedelta(days=30)  # TTL 정책 설정
run.log_artifact(artifact)
```

앞의 코드 조각은 아티팩트의 TTL 정책을 30일로 설정합니다. 즉, W&B는 30일 후에 아티팩트를 삭제합니다.

### 아티팩트를 생성한 후 TTL 정책 설정 또는 편집하기
W&B 앱 UI 또는 W&B Python SDK를 사용하여 이미 존재하는 아티팩트에 대한 TTL 정책을 정의합니다.

:::note
아티팩트의 TTL을 수정할 때, 아티팩트가 만료되는 시간은 여전히 아티팩트의 `createdAt` 타임스탬프를 사용하여 계산됩니다.
:::

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python SDK', value: 'python'},
    {label: 'W&B 앱', value: 'app'},
  ]}>
  <TabItem value="python">

1. [아티팩트 가져오기](./download-and-use-an-artifact.md).
2. 아티팩트의 `ttl` 속성에 시간 차이를 전달합니다.
3. [`save`](../../ref/python/run.md#save) 메서드로 아티팩트를 업데이트합니다.


다음 코드 조각은 아티팩트에 대한 TTL 정책을 설정하는 방법을 보여줍니다:
```python
import wandb
from datetime import timedelta

artifact = run.use_artifact("<my-entity/my-project/my-artifact:alias>")
artifact.ttl = timedelta(days=365 * 2)  # 2년 후 삭제
artifact.save()
```

앞의 코드 예제는 TTL 정책을 2년으로 설정합니다.

  </TabItem>
  <TabItem value="app">

1. W&B 앱 UI에서 W&B 프로젝트로 이동합니다.
2. 왼쪽 패널에서 아티팩트 아이콘을 선택합니다.
3. 아티팩트 목록에서 확장하고자 하는 아티팩트 유형을 선택합니다.
4. TTL 정책을 편집하려는 아티팩트 버전을 선택합니다.
5. **버전** 탭을 클릭합니다.
6. 드롭다운에서 **TTL 정책 편집**을 선택합니다.
7. 나타나는 모달에서 TTL 정책 드롭다운에서 **사용자 정의**를 선택합니다.
8. **TTL 지속 시간** 필드에서 일 단위로 TTL 정책을 설정합니다.
9. **TTL 업데이트** 버튼을 선택하여 변경 사항을 저장합니다.

![](/images/artifacts/edit_ttl_ui.gif)

  </TabItem>
</Tabs>

### 팀의 기본 TTL 정책 설정하기

:::info
팀 관리자만 팀의 기본 TTL 정책을 설정할 수 있습니다.
:::

팀의 기본 TTL 정책을 설정합니다. 기본 TTL 정책은 각각의 생성 날짜에 따라 모든 기존 및 미래의 아티팩트에 적용됩니다. 기존 버전 수준의 TTL 정책이 있는 아티팩트는 팀의 기본 TTL에 영향을 받지 않습니다.

1. 팀의 프로필 페이지로 이동합니다.
2. **설정** 탭을 선택합니다.
3. **아티팩트 수명 시간 (TTL) 섹션**으로 이동합니다.
4. **팀의 기본 TTL 정책 설정**을 클릭합니다.
5. **기간** 필드에서 일 단위로 TTL 정책을 설정합니다.
6. **설정 검토 및 저장**을 클릭합니다.
7. 변경 사항을 확인한 다음 **설정 저장**을 선택합니다.

![](/images/artifacts/set_default_ttl.gif)

## TTL 정책 비활성화하기
W&B Python SDK 또는 W&B 앱 UI를 사용하여 특정 아티팩트 버전에 대한 TTL 정책을 비활성화합니다.


<Tabs
  defaultValue="python"
  values={[
    {label: 'Python SDK', value: 'python'},
    {label: 'W&B 앱', value: 'app'},
  ]}>
  <TabItem value="python">

1. [아티팩트 가져오기](./download-and-use-an-artifact.md).
2. 아티팩트의 `ttl` 속성을 `None`으로 설정합니다.
3. [`save`](../../ref/python/run.md#save) 메서드로 아티팩트를 업데이트합니다.


다음 코드 조각은 아티팩트에 대한 TTL 정책을 비활성화하는 방법을 보여줍니다:
```python
artifact = run.use_artifact("<my-entity/my-project/my-artifact:alias>")
artifact.ttl = None
artifact.save()
```


  </TabItem>
  <TabItem value="app">

1. W&B 앱 UI에서 W&B 프로젝트로 이동합니다.
2. 왼쪽 패널에서 아티팩트 아이콘을 선택합니다.
3. 아티팩트 목록에서 확장하고자 하는 아티팩트 유형을 선택합니다.
4. TTL 정책을 편집하려는 아티팩트 버전을 선택합니다.
5. 버전 탭을 클릭합니다.
6. **레지스트리에 링크** 버튼 옆의 미트볼 UI 아이콘을 클릭합니다.
7. 드롭다운에서 **TTL 정책 편집**을 선택합니다.
8. 나타나는 모달에서 TTL 정책 드롭다운에서 **비활성화**를 선택합니다.
9. **TTL 업데이트** 버튼을 선택하여 변경 사항을 저장합니다.

![](/images/artifacts/remove_ttl_polilcy.gif)

  </TabItem>
</Tabs>

## TTL 정책 보기
Python SDK 또는 W&B 앱 UI로 아티팩트의 TTL 정책을 확인합니다.

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python SDK', value: 'python'},
    {label: 'W&B 앱', value: 'app'},
  ]}>
  <TabItem value="python">

아티팩트의 TTL 정책을 보려면 print 문을 사용하세요. 다음 예제는 아티팩트를 검색하고 그것의 TTL 정책을 보는 방법을 보여줍니다:

```python
artifact = run.use_artifact("<my-entity/my-project/my-artifact:alias>")
print(artifact.ttl)
```

  </TabItem>
  <TabItem value="app">


W&B 앱 UI로 아티팩트의 TTL 정책을 확인하세요.

1. [https://wandb.ai](https://wandb.ai)에서 W&B 앱으로 이동합니다.
2. W&B 프로젝트로 이동합니다.
3. 프로젝트 내에서 왼쪽 사이드바의 Artifacts 탭을 선택합니다.
4. 컬렉션을 클릭합니다.

선택한 컬렉션의 모든 아티팩트를 볼 수 있는 컬렉션 뷰에서 `Time to Live` 열에서 해당 아티팩트에 할당된 TTL 정책을 볼 수 있습니다.

![](/images/artifacts/ttl_collection_panel_ui.png)

  </TabItem>
</Tabs>