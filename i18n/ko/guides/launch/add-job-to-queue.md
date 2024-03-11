---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 작업 대기열에 작업 추가하기

다음 페이지에서는 작업 대기열에 실행 작업을 추가하는 방법을 설명합니다.

:::info
당신 또는 당신의 팀 멤버가 이미 작업 대기열을 설정했는지 확인하세요. 자세한 내용은 [Launch 설정](./setup-launch.md) 페이지를 참조하세요.
:::

## 대기열에 작업 추가하기

W&B 앱을 사용하여 대화형으로 또는 W&B CLI를 사용하여 프로그래매틱하게 대기열에 작업을 추가하세요.

<Tabs
  defaultValue="app"
  values={[
    {label: 'W&B 앱', value: 'app'},
    {label: 'W&B CLI', value: 'cli'},
  ]}>
  <TabItem value="app">
W&B 앱을 사용하여 프로그래매틱하게 대기열에 작업을 추가하세요.

1. W&B 프로젝트 페이지로 이동합니다.
2. 왼쪽 패널에서 **작업** 아이콘을 선택하세요:
  ![](/images/launch/project_jobs_tab_gs.png)
3. **작업** 페이지에는 이전에 실행된 W&B run에서 생성된 W&B launch 작업 목록이 표시됩니다. 
  ![](/images/launch/view_jobs.png)
4. 작업 이름 옆에 있는 **Launch** 버튼을 선택하세요. 모달이 페이지의 오른쪽에 나타납니다.
5. **작업 버전** 드롭다운에서 사용하려는 launch 작업의 버전을 선택하세요. Launch 작업은 다른 [W&B 아티팩트](../artifacts/create-a-new-artifact-version.md)처럼 버전 관리됩니다. 소프트웨어 의존성이나 작업을 실행하는 데 사용된 소스 코드를 수정하면 동일한 launch 작업의 다른 버전이 생성됩니다.
6. **Overrides** 섹션에서 launch 작업에 대해 구성된 입력값에 대한 새로운 값을 제공하세요. 일반적인 오버라이드에는 새로운 엔트리포인트 코맨드, 인수 또는 새로운 W&B run의 `wandb.config`에서의 값이 포함됩니다.  
  ![](/images/launch/create_starter_queue_gs.png)
  **Paste from...** 버튼을 클릭하여 launch 작업을 사용한 다른 W&B run에서 값을 복사하여 붙여넣을 수 있습니다.
7. **Queue** 드롭다운에서 launch 작업을 추가하려는 launch 대기열의 이름을 선택하세요.
8. **작업 우선순위** 드롭다운을 사용하여 launch 작업의 우선순위를 지정하세요. launch 대기열이 우선순위 지정을 지원하지 않는 경우 launch 작업의 우선순위는 "중간"으로 설정됩니다.
9. **(선택 사항) 팀 관리자가 큐 구성 템플릿을 생성한 경우에만 이 단계를 따르세요**  
**Queue Configurations** 필드 내에서 팀 관리자가 생성한 구성 옵션에 대한 값을 제공하세요.  
예를 들어, 다음 예제에서 팀 관리자는 팀이 사용할 수 있는 AWS 인스턴스 유형을 구성했습니다. 이 경우, 팀 멤버는 모델을 훈련시키기 위해 `ml.m4.xlarge` 또는 `ml.p3.xlarge` 컴퓨트 인스턴스 유형을 선택할 수 있습니다.
![](/images/launch/team_member_use_config_template.png)
10. 결과 run이 표시될 **목적지 프로젝트**를 선택하세요. 이 프로젝트는 대기열과 동일한 엔티티에 속해야 합니다.
11. **Launch now** 버튼을 선택하세요.


  </TabItem>
    <TabItem value="cli">

`wandb launch` 코맨드를 사용하여 대기열에 작업을 추가하세요. 하이퍼파라미터 오버라이드가 포함된 JSON 구성을 생성하세요. 예를 들어, [퀵스타트](./walkthrough.md) 가이드에서의 스크립트를 사용하여 다음 오버라이드가 포함된 JSON 파일을 생성합니다:

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
JSON 구성 파일을 제공하지 않는 경우 W&B Launch는 기본 파라미터를 사용할 것입니다.
:::

대기열 구성을 오버라이드하거나 대기열에 구성 리소스가 정의되어 있지 않은 경우, config.json 파일에 `resource_args` 키를 지정할 수 있습니다. 예를 들어, 위 예제를 계속하여, config.json 파일은 다음과 같이 보일 수 있습니다:

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



`queue`(`-q`) 플래그를 위한 대기열 이름, `job`(`-j`) 플래그를 위한 작업 이름, 그리고 `config`(`-c`) 플래그를 위한 구성 파일의 경로를 제공하세요.

```bash
wandb launch -j <job> -q <queue-name> \ 
-e <entity-name> -c path/to/config.json
```
W&B 팀 내에서 작업하는 경우, `entity` 플래그 (`-e`)를 지정하여 대기열이 사용할 엔티티를 나타내는 것이 좋습니다.

  </TabItem>
</Tabs>