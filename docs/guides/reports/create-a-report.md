---
title: Create a report
description: W&B 리포트를 App UI로 생성하거나 Weights & Biases SDK를 사용하여 프로그래밍 방식으로 생성합니다.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

리포트를 W&B App UI를 사용하여 대화식으로 생성하거나 W&B Python SDK를 사용하여 프로그래밍 방식으로 생성하세요.

:::info
예제를 보려면 [Google Colab을 참고하세요](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Report_API_Quickstart.ipynb).
:::

<Tabs
  defaultValue="app"
  values={[
    {label: 'App UI', value: 'app'},
    {label: 'Report tab', value: 'reporttab'},
    {label: 'W&B Python SDK', value: 'sdk'},
  ]}>
  <TabItem value="app">

1. W&B App에서 프로젝트 워크스페이스로 이동합니다.
2. 워크스페이스의 오른쪽 상단에서 **Create report**를 클릭합니다.

![](/images/reports/create_a_report_button.png)

3. 모달이 나타납니다. 시작하고 싶은 차트를 선택합니다. 나중에 리포트 인터페이스에서 차트를 추가하거나 삭제할 수 있습니다.

![](/images/reports/create_a_report_modal.png)

4. **Filter run sets** 옵션을 선택하여 새로운 run들이 리포트에 추가되지 않도록 합니다. 이 옵션은 켜거나 끌 수 있습니다. **Create report**를 클릭하면 리포트 탭에서 계속 작업할 수 있는 임시 리포트가 제공됩니다.

  </TabItem>
  <TabItem value="reporttab">

1. W&B App에서 프로젝트 워크스페이스로 이동합니다.
2. 프로젝트에서 **Reports** 탭(클립보드 이미지)을 선택합니다.
3. 리포트 페이지에서 **Create Report** 버튼을 선택합니다.

![](/images/reports/create_report_button.png)
  </TabItem>
  <TabItem value="sdk">

`wandb` 라이브러리를 사용하여 프로그래밍 방식으로 리포트를 생성합니다. 

1. W&B SDK와 Workspaces API를 설치합니다:
```bash
pip install wandb wandb-workspaces
```
2. 다음으로, 워크스페이스를 가져옵니다:
```python
import wandb
import wandb_workspaces.reports.v2 as wr
```
3. `wandb_workspaces.reports.v2.Report`를 사용하여 리포트를 생성합니다. Report 클래스의 Public API([`wandb.apis.reports`](/ref/python/public-api/api#reports))를 사용하여 리포트 인스턴스를 생성합니다. 프로젝트 이름을 지정하세요.

```python
report = wr.Report(project="report_standard")
```

4. 리포트를 저장합니다. 리포트는 `.save()` 메소드를 호출하기 전까지 W&B 서버에 업로드되지 않습니다:

```python
report.save()
```

App UI를 사용하여 대화식으로 리포트를 편집하거나 프로그래밍 방식으로 편집하는 방법에 대한 정보는 [Edit a report](/guides/reports/edit-a-report)를 참조하세요.
  </TabItem>
</Tabs>