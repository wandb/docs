---
title: Add job to queue
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

다음 페이지는 launch 큐에 launch 작업을 추가하는 방법을 설명합니다.

:::info
당신이나 팀의 다른 사람이 이미 launch 큐를 설정했는지 확인하세요. 자세한 내용은 [Set up Launch](./setup-launch.md) 페이지를 참조하세요.
:::

## 큐에 작업 추가하기

W&B 앱 또는 W&B CLI로 큐에 인터랙티브하게 작업을 추가하세요.

<Tabs
  defaultValue="app"
  values={[
    {label: 'W&B App', value: 'app'},
    {label: 'W&B CLI', value: 'cli'},
  ]}>
  <TabItem value="app">
W&B 앱을 사용하여 프로그래매틱하게 큐에 작업을 추가합니다.

1. W&B 프로젝트 페이지로 이동합니다.
2. 왼쪽 패널에서 **Jobs** 아이콘을 선택합니다:
  ![](/images/launch/project_jobs_tab_gs.png)
3. **Jobs** 페이지에는 이전에 실행된 W&B runs에서 생성된 W&B launch jobs의 목록이 표시됩니다. 
  ![](/images/launch/view_jobs.png)
4. 작업 이름 옆의 **Launch** 버튼을 선택합니다. 페이지 오른쪽에 모달이 나타납니다.
5. **Job version** 드롭다운에서 사용하려는 launch job의 버전을 선택합니다. Launch jobs는 다른 [W&B Artifact](../artifacts/create-a-new-artifact-version.md)처럼 versioning 됩니다. Software dependencies 또는 작업 실행에 사용되는 source code를 수정하면 동일한 launch job의 다른 버전이 생성됩니다.
6. **Overrides** 섹션 내에서 launch job에 대해 설정된 입력값에 대한 새로운 값을 제공합니다. 일반적인 overrides로는 새로운 entrypoint 명령, 인수, 또는 새로운 W&B run의 `wandb.config` 내 값이 포함됩니다.  
  ![](/images/launch/create_starter_queue_gs.png)
  **Paste from...** 버튼을 클릭하여 launch job을 사용한 다른 W&B runs에서 값을 복사하여 붙여넣을 수 있습니다.
7. **Queue** 드롭다운에서 launch job을 추가하려는 launch 큐의 이름을 선택합니다.
8. **Job Priority** 드롭다운을 사용하여 launch job의 우선순위를 지정합니다. Launch 큐가 우선순위 지정을 지원하지 않는 경우 launch job의 우선순위는 "Medium"으로 설정됩니다.
9. **(선택 사항) 팀 관리자가 큐 설정 템플릿을 생성한 경우에만 이 단계를 따르세요**  
**Queue Configurations** 필드 내에서 팀의 관리자가 생성한 설정 옵션에 대한 값을 제공합니다.  
예를 들어, 다음 예에서는 팀 관리자가 팀이 사용할 수 있도록 AWS 인스턴스 유형을 설정했습니다. 이 경우, 팀 멤버는 `ml.m4.xlarge` 또는 `ml.p3.xlarge` 컴퓨팅 인스턴스 유형 중 하나를 선택하여 모델을 학습시킬 수 있습니다.
![](/images/launch/team_member_use_config_template.png)
10. 결과 run이 표시될 **Destination project**를 선택합니다. 이 프로젝트는 큐와 동일한 entity에 속해야 합니다.
11. **Launch now** 버튼을 선택합니다.


  </TabItem>
    <TabItem value="cli">

`wandb launch` 코맨드를 사용하여 큐에 작업을 추가합니다. 하이퍼파라미터 overrides가 포함된 JSON 설정 파일을 만듭니다. 예를 들어, [Quickstart](./walkthrough.md) 가이드의 스크립트를 사용하여 다음과 같은 overrides가 포함된 JSON 파일을 생성합니다:

```json title="config.json"
{
  "overrides": {
      "args": [],
      "run_config": {
          "learning_rate": 0,
          "epochs": 0
      },   
      "entry_point": []
  }
}
```

:::note
JSON 설정 파일을 제공하지 않으면 W&B Launch는 기본 파라미터를 사용합니다.
:::

큐 설정을 override하거나, launch 큐에 정의된 설정 리소스가 없는 경우, `config.json` 파일에서 `resource_args` 키를 지정할 수 있습니다. 예를 들어, 위의 예를 계속하여, `config.json` 파일은 다음과 유사하게 보일 수 있습니다:

```json title="config.json"
{
  "overrides": {
      "args": [],
      "run_config": {
          "learning_rate": 0,
          "epochs": 0
      },
      "entry_point": []
  },
  "resource_args": {
        "<resource-type>" : {
            "<key>": "<value>"
        }
  }
}
```

`<>` 내의 값을 자신의 값으로 대체하세요.

`queue`(`-q`) 플래그에는 큐의 이름을, `job`(`-j`) 플래그에는 작업의 이름을, 그리고 설정 파일의 경로는 `config`(`-c`) 플래그로 제공합니다.

```bash
wandb launch -j <job> -q <queue-name> \ 
-e <entity-name> -c path/to/config.json
```
W&B Team에서 작업하는 경우, 큐가 사용할 entity를 나타내기 위해 `entity` 플래그 (`-e`)를 지정하는 것이 좋습니다.

  </TabItem>
</Tabs>