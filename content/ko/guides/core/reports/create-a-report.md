---
title: 리포트 생성
description: W&B App을 사용하거나 프로그래밍 방식으로 W&B Report를 생성하세요.
menu:
  default:
    identifier: ko-guides-core-reports-create-a-report
    parent: reports
weight: 10
---

{{% alert %}}
W&B Report 및 Workspace API는 Public Preview 단계에 있습니다.
{{% /alert %}}

아래 탭을 선택하여 W&B App에서 또는 W&B Report 및 Workspace API를 사용해 프로그래밍 방식으로 report를 생성하는 방법을 알아보세요.

프로그램적으로 report를 생성하는 예시는 [Google Colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Report_API_Quickstart.ipynb)에서 확인할 수 있습니다.

{{< tabpane text=true >}}
{{% tab header="W&B App" value="app" %}}
1. W&B App에서 본인의 project workspace로 이동하세요.
2. 워크스페이스 오른쪽 상단에서 **Create report** 버튼을 클릭하세요.

   {{< img src="/images/reports/create_a_report_button.png" alt="Create report button" >}}

3. 모달 창이 나타납니다. 시작할 차트를 선택할 수 있으며, 이후에 report 인터페이스에서 차트를 추가하거나 삭제할 수도 있습니다.

    {{< img src="/images/reports/create_a_report_modal.png" alt="Create report modal" >}}

4. 새로운 run이 report에 추가되는 것을 방지하려면 **Filter run sets** 옵션을 선택하세요. 이 옵션은 켜거나 끌 수 있습니다. **Create report**를 클릭하면, draft 상태의 report가 report 탭에 생성되어 계속 작업할 수 있습니다.
{{% /tab %}}

{{% tab header="Report tab" value="reporttab"%}}
1. W&B App에서 본인의 project workspace로 이동하세요.
2. 프로젝트 내 **Reports** 탭(클립보드 이미지)을 선택하세요.
3. report 페이지에서 **Create Report** 버튼을 클릭하세요.

   {{< img src="/images/reports/create_report_button.png" alt="Create report button" >}}
{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api"%}}
프로그램적으로 report를 생성하는 방법:

1. W&B SDK(`wandb`) 및 Report and Workspace API(`wandb-workspaces`)를 설치하세요.
    ```bash
    pip install wandb wandb-workspaces
    ```
2. 다음으로 workspaces를 import하세요.
    ```python
    import wandb
    import wandb_workspaces.reports.v2 as wr
    ```
3. `wandb_workspaces.reports.v2.Report`로 report를 생성하세요. Report Class Public API([`wandb.apis.reports`]({{< relref path="/ref/python/public-api/api.md#reports" lang="ko" >}}))를 사용하여 인스턴스를 만듭니다. project 이름을 지정하세요.
    ```python
    report = wr.Report(project="report_standard")
    ```
4. report를 저장하세요. Reports는 .`save()` 메소드를 호출하기 전까지 W&B 서버에 업로드되지 않습니다.
    ```python
    report.save()
    ```

App UI에서 인터랙티브하게 report를 수정하거나 프로그램적으로 편집하는 방법은 [Edit a report]({{< relref path="/guides/core/reports/edit-a-report" lang="ko" >}})에서 확인할 수 있습니다.
{{% /tab %}}
{{< /tabpane >}}