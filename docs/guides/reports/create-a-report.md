import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Create a report

Create a Report interactively with the App UI or programmatically with the `wandb` Python SDK.

:::warning
Creating Reports programmatically with the Python SDK is in Beta and in active development.
:::

<Tabs
  defaultValue="app"
  values={[
    {label: 'App UI', value: 'app'},
    {label: 'Report tab', value: 'reporttab'},
    {label: 'Python SDK (Beta)', value: 'sdk'},
  ]}>
  <TabItem value="app">

Click **Create report** in the upper right corner of your workspace.

![](<pathname:///images/reports/create_a_report_button.png>)

Select the charts you would like to start with. You can add or delete charts later from the report interface.

![](<pathname:///images/reports/create_a_report_modal.png>)

Select the **Filter run sets** option to prevent new runs from being added to your report. You can toggle this option on or off. Once you click **Create report,** a draft report will be available in the report tab to continue working on.
  </TabItem>
  <TabItem value="reporttab">

Navigate to the **Reports** tab in your project and select the **Create Report** button on the report page. This creates a new blank report. Save a report to get a shareable link, or send charts to the report from different workspaces, and different projects.

![](<../../.gitbook/assets/image (180).png>)
  </TabItem>
  <TabItem value="sdk">

Create a report programmatically with the `wandb` library. After you import the `wandb`, state `wandb.require('report-editing')` to enable programatic report editing. This requirement ensures you do not accidentally modify a report.

```python
import wandb
import wandb.apis.reports as wr

# W&B requirement to avoid accidental report modification
wandb.require('report-editing')
```

Create a report instance with the Report Class Public API ([`wandb.apis.reports`](https://docs.wandb.ai/ref/python/public-api/api#reports)). Specify a name for the project.

```python
report = wr.Report(project='report_standard')
```

Reports are not uploaded to the Weights & Biases server until you call the .`save()` method:

```python
report.save()
```

For information on how to edit a report interactively with the App UI or programmatically, see [Edit a report](https://docs.wandb.ai/guides/reports/edit-a-report).
  </TabItem>
</Tabs>