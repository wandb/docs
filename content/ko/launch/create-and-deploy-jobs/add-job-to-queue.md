---
title: Add job to queue
menu:
  launch:
    identifier: ko-launch-create-and-deploy-jobs-add-job-to-queue
    parent: create-and-deploy-jobs
url: /ko/guides//launch/add-job-to-queue
---

다음 페이지에서는 launch queue에 launch job을 추가하는 방법을 설명합니다.

{{% alert %}}
귀하 또는 귀하의 팀 구성원 중 한 명이 이미 launch queue를 구성했는지 확인하십시오. 자세한 내용은 [Launch 설정]({{< relref path="/launch/set-up-launch/" lang="ko" >}}) 페이지를 참조하십시오.
{{% /alert %}}

## queue에 job 추가

W&B App을 사용하여 대화형으로 또는 W&B CLI를 사용하여 프로그래밍 방식으로 queue에 job을 추가합니다.

{{< tabpane text=true >}}
{{% tab "W&B app" %}}
W&B App을 사용하여 프로그래밍 방식으로 queue에 job을 추가합니다.

1. W&B Project 페이지로 이동합니다.
2. 왼쪽 패널에서 **Jobs** 아이콘을 선택합니다:
  {{< img src="/images/launch/project_jobs_tab_gs.png" alt="" >}}
3. **Jobs** 페이지에는 이전에 실행된 W&B run에서 생성된 W&B launch job 목록이 표시됩니다.
  {{< img src="/images/launch/view_jobs.png" alt="" >}}
4. Job 이름 옆에 있는 **Launch** 버튼을 선택합니다. 페이지 오른쪽에 모달이 나타납니다.
5. **Job version** 드롭다운에서 사용하려는 launch job 버전을 선택합니다. Launch job은 다른 [W&B Artifact]({{< relref path="/guides/core/artifacts/create-a-new-artifact-version.md" lang="ko" >}})처럼 버전이 관리됩니다. job을 실행하는 데 사용되는 소프트웨어 종속성 또는 소스 코드를 수정하면 동일한 launch job의 다른 버전이 생성됩니다.
6. **Overrides** 섹션 내에서 launch job에 대해 구성된 모든 입력에 대해 새 값을 제공합니다. 일반적인 재정의에는 새 진입점 커맨드, 인수 또는 새 W&B run의 `wandb.config`의 값이 포함됩니다.
  {{< img src="/images/launch/create_starter_queue_gs.png" alt="" >}}
  **Paste from...** 버튼을 클릭하여 launch job을 사용한 다른 W&B run에서 값을 복사하여 붙여넣을 수 있습니다.
7. **Queue** 드롭다운에서 launch job을 추가할 launch queue 이름을 선택합니다.
8. **Job Priority** 드롭다운을 사용하여 launch job의 우선 순위를 지정합니다. Launch queue가 우선 순위 지정을 지원하지 않는 경우 launch job의 우선 순위는 "Medium"으로 설정됩니다.
9. **(선택 사항) 팀 관리자가 queue 설정 템플릿을 만든 경우에만 이 단계를 따르십시오.**
**Queue Configurations** 필드 내에서 팀 관리자가 만든 구성 옵션에 대한 값을 제공합니다.
예를 들어 다음 예에서 팀 관리자는 팀에서 사용할 수 있는 AWS 인스턴스 유형을 구성했습니다. 이 경우 팀 멤버는 `ml.m4.xlarge` 또는 `ml.p3.xlarge` 컴퓨팅 인스턴스 유형을 선택하여 모델을 학습할 수 있습니다.
{{< img src="/images/launch/team_member_use_config_template.png" alt="" >}}
10. 결과 run이 표시될 **Destination project**를 선택합니다. 이 프로젝트는 queue와 동일한 entity에 속해야 합니다.
11. **Launch now** 버튼을 선택합니다.

{{% /tab %}}
{{% tab "W&B CLI" %}}

`wandb launch` 커맨드를 사용하여 queue에 job을 추가합니다. 하이퍼파라미터 재정의를 사용하여 JSON 설정을 만듭니다. 예를 들어 [퀵스타트]({{< relref path="../walkthrough.md" lang="ko" >}}) 가이드의 스크립트를 사용하여 다음 재정의를 사용하여 JSON 파일을 만듭니다.

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

{{% alert %}}
JSON 구성 파일을 제공하지 않으면 W&B Launch가 기본 파라미터를 사용합니다.
{{% /alert %}}

queue 구성을 재정의하려는 경우 또는 launch queue에 구성 리소스가 정의되지 않은 경우 config.json 파일에서 `resource_args` 키를 지정할 수 있습니다. 예를 들어 위의 예제를 계속 따르면 config.json 파일이 다음과 유사하게 보일 수 있습니다.

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

`<>` 안의 값을 자신의 값으로 바꿉니다.

`queue`(`-q`) 플래그에 대한 queue 이름, `job`(`-j`) 플래그에 대한 job 이름, `config`(`-c`) 플래그에 대한 구성 파일의 경로를 제공합니다.

```bash
wandb launch -j <job> -q <queue-name> \ 
-e <entity-name> -c path/to/config.json
```
W&B Team 내에서 작업하는 경우 queue가 사용할 entity를 나타내기 위해 `entity` 플래그(`-e`)를 지정하는 것이 좋습니다.

{{% /tab %}}
{{% /tabpane %}}
