---
title: Create a report
description: Weights & Biases SDK를 사용하여 앱 UI 또는 프로그래밍 방식으로 W&B **Report**를 만드세요.
menu:
  default:
    identifier: ko-guides-core-reports-create-a-report
    parent: reports
weight: 10
---

W&B App UI 또는 W&B Python SDK를 사용하여 프로그래밍 방식으로 리포트를 상호 작용하며 생성합니다.

{{% alert %}}
[예제 Google Colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Report_API_Quickstart.ipynb)을 참조하세요.
{{% /alert %}}

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}
1. W&B App에서 프로젝트 워크스페이스로 이동합니다.
2. 워크스페이스의 오른쪽 상단에서 **Create report**를 클릭합니다.

   {{< img src="/images/reports/create_a_report_button.png" alt="" >}}

3. 모달이 나타납니다. 시작하려는 차트를 선택합니다. 리포트 인터페이스에서 나중에 차트를 추가하거나 삭제할 수 있습니다.

    {{< img src="/images/reports/create_a_report_modal.png" alt="" >}}

4. 새로운 run이 리포트에 추가되는 것을 방지하려면 **Filter run sets** 옵션을 선택합니다. 이 옵션을 켜거나 끌 수 있습니다. **Create report**를 클릭하면 리포트 탭에서 계속 작업할 수 있는 초안 리포트를 사용할 수 있습니다.
{{% /tab %}}

{{% tab header="Report tab" value="reporttab"%}}
1. W&B App에서 프로젝트 워크스페이스로 이동합니다.
2. 프로젝트에서 **Reports** 탭(클립보드 이미지)을 선택합니다.
3. 리포트 페이지에서 **Create Report** 버튼을 선택합니다.

   {{< img src="/images/reports/create_report_button.png" alt="" >}}
{{% /tab %}}

{{% tab header="W&B Python SDK" value="sdk"%}}
`wandb` 라이브러리를 사용하여 프로그래밍 방식으로 리포트를 생성합니다.

1. W&B SDK 및 Workspaces API를 설치합니다:
    ```bash
    pip install wandb wandb-workspaces
    ```
2. 다음으로, 워크스페이스를 가져옵니다.
    ```python
    import wandb
    import wandb_workspaces.reports.v2 as wr
    ```       
3. `wandb_workspaces.reports.v2.Report`를 사용하여 리포트를 생성합니다. Report Class Public API([`wandb.apis.reports`]({{< relref path="/ref/python/public-api/api#reports" lang="ko" >}}))를 사용하여 리포트 인스턴스를 생성합니다. 프로젝트 이름을 지정합니다.
    ```python
    report = wr.Report(project="report_standard")
    ```

4. 리포트를 저장합니다. `.save()` 메서드를 호출할 때까지 리포트가 W&B 서버에 업로드되지 않습니다.
    ```python
    report.save()
    ```

App UI 또는 프로그래밍 방식으로 리포트를 상호 작용하며 편집하는 방법에 대한 자세한 내용은 [리포트 편집]({{< relref path="/guides/core/reports/edit-a-report" lang="ko" >}})을 참조하십시오.
{{% /tab %}}
{{< /tabpane >}}
