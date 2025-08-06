---
title: 큐에 작업 추가
menu:
  launch:
    identifier: ko-launch-create-and-deploy-jobs-add-job-to-queue
    parent: create-and-deploy-jobs
url: guides/launch/add-job-to-queue
---

다음 페이지에서는 launch queue 에 launch job 을 추가하는 방법을 설명합니다.

{{% alert %}}
여러분 또는 팀원이 이미 launch queue 를 설정했는지 확인하세요. 자세한 내용은 [Set up Launch]({{< relref path="/launch/set-up-launch/" lang="ko" >}}) 페이지를 참조하세요.
{{% /alert %}}

## 큐에 job 추가하기

W&B App 을 사용하여 대화형으로, 또는 W&B CLI 를 사용하여 프로그래밍 방식으로 큐에 job 을 추가할 수 있습니다.

{{< tabpane text=true >}}
{{% tab "W&B app" %}}
W&B App 을 통해 프로그램적으로 queue 에 job 을 추가하세요.

1. 여러분의 W&B Project 페이지로 이동하세요.
2. 왼쪽 패널에서 **Jobs** 아이콘을 선택하세요:
  {{< img src="/images/launch/project_jobs_tab_gs.png" alt="Project Jobs tab" >}}
3. **Jobs** 페이지에서는 이전에 실행된 W&B run 에서 생성된 W&B launch job 들의 목록을 볼 수 있습니다. 
  {{< img src="/images/launch/view_jobs.png" alt="Jobs listing" >}}
4. 추가하고 싶은 Job 이름 옆의 **Launch** 버튼을 선택하세요. 페이지 오른쪽에 모달이 나타납니다.
5. **Job version** 드롭다운에서 사용할 launch job 의 버전을 선택하세요. Launch job 은 다른 [W&B Artifact]({{< relref path="/guides/core/artifacts/create-a-new-artifact-version.md" lang="ko" >}}) 와 마찬가지로 버전 관리가 가능합니다. 소프트웨어 의존성이나 job 실행에 사용하는 소스 코드를 변경하면 같은 launch job 의 새로운 버전이 생성됩니다.
6. **Overrides** 섹션에서, launch job 에 대해 구성된 입력값 중 새로 지정하고 싶은 값을 입력하세요. 보통 새로운 entrypoint 명령어, 인수, 또는 새로운 W&B run 의 `wandb.Run.config` 파라미터 값 등이 여기에 포함됩니다.  
  {{< img src="/images/launch/create_starter_queue_gs.png" alt="Queue configuration" >}}
  **Paste from...** 버튼을 클릭해서 같은 launch job 을 사용한 다른 W&B run 의 값을 복사/붙여넣기 할 수 있습니다.
7. **Queue** 드롭다운에서, 이 launch job 을 추가하고자 하는 launch queue 이름을 선택하세요. 
8. **Job Priority** 드롭다운을 사용하여 launch job 의 우선순위를 지정하세요. launch queue 에서 우선순위 기능을 지원하지 않는 경우 기본값은 "Medium" 으로 설정됩니다.
9. **(선택) 이 단계는 팀 관리자에 의해 queue config template 이 생성된 경우에만 따라 하세요**  
**Queue Configurations** 필드에서, 팀 관리자가 생성한 설정 옵션에 대한 값을 입력하세요.  
예를 들어, 아래 예시에서는 팀 관리자가 팀에서 사용할 수 있는 AWS 인스턴스 타입을 미리 설정해두었습니다. 이 경우 팀원들은 `ml.m4.xlarge` 또는 `ml.p3.xlarge` 컴퓨트 인스턴스 타입 중 하나를 선택해서 모델 학습에 사용할 수 있습니다.
{{< img src="/images/launch/team_member_use_config_template.png" alt="Config template selection" >}}
10. 실행 결과인 run 이 표시될 **Destination project** 를 선택하세요. 이 프로젝트는 queue 와 동일한 entity 에 속해야 합니다.
11. **Launch now** 버튼을 선택하세요. 

{{% /tab %}}
{{% tab "W&B CLI" %}}

`wandb launch` 명령어를 사용하여 job 을 queue 에 추가할 수 있습니다. 하이퍼파라미터 override 값을 포함한 JSON 설정 파일을 만드세요. 예를 들어 [Quickstart]({{< relref path="../walkthrough.md" lang="ko" >}}) 가이드의 스크립트를 사용할 때 아래와 같은 override 가 포함된 JSON 파일을 생성합니다:

```json title="config.json"
// 하이퍼파라미터와 인수를 오버라이드 합니다.
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

{{% alert %}}
JSON 설정 파일을 제공하지 않으면 W&B Launch 는 기본 파라미터를 사용합니다.
{{% /alert %}}

queue 설정을 오버라이드하거나, launch queue 에 설정 리소스가 정의되어 있지 않은 경우에는 config.json 파일에서 `resource_args` 키를 직접 지정할 수 있습니다. 예를 들어 위의 예시에서 config.json 파일이 다음과 비슷하게 될 수 있습니다:

```json title="config.json"
// 리소스별 추가 설정을 명시합니다.
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

`<>` 안의 값들은 여러분의 설정에 맞게 바꿔주세요.

`queue`(`-q`) 플래그에는 queue 이름을, `job`(`-j`) 플래그에는 job 이름을, `config`(`-c`) 플래그에는 설정 파일 경로를 지정하세요.

```bash
wandb launch -j <job> -q <queue-name> \ 
-e <entity-name> -c path/to/config.json
```
W&B 팀에서 작업하신다면, 어떤 entity 가 queue 를 사용하는지 지정할 수 있도록 `entity` 플래그 (`-e`) 사용을 권장합니다.

{{% /tab %}}
{{% /tabpane %}}