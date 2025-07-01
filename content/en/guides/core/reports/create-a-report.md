---
description: Create a W&B Report with the W&B App or programmatically.
menu:
  default:
    identifier: create-a-report
    parent: reports
title: Create a report
weight: 10
---

{{% alert %}}
W&B Report and Workspace API is in Public Preview.
{{% /alert %}}

Select a tab below to learn how to create a report in the W&B App or programmatically with the W&B Report and Workspace API.

See this [Google Colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Report_API_Quickstart.ipynb) for an example on how to programmatically create a report.


{{< tabpane text=true >}}
{{% tab header="W&B App" value="app" %}}
1. Navigate to your project workspace in the W&B App.
2. Click **Create report** in the upper right corner of your workspace.

   {{< img src="/images/reports/create_a_report_button.png" alt="Create a report button in W&B workspace showing report generation options" >}}

3. A modal will appear. Select the charts you would like to start with. You can add or delete charts later from the report interface.

    {{< img src="/images/reports/create_a_report_modal.png" alt="Create report modal dialog showing template options and report configuration settings" >}}

4. Select the **Filter run sets** option to prevent new runs from being added to your report. You can toggle this option on or off. Once you click **Create report,** a draft report will be available in the report tab to continue working on.
{{% /tab %}}

{{% tab header="Report tab" value="reporttab"%}}
1. Navigate to your project workspace in the W&B App.
2. Select to the **Reports** tab (clipboard image) in your project.
3. Select the **Create Report** button on the report page. 

   {{< img src="/images/reports/create_report_button.png" alt="Create report button interface showing available chart options for report creation" >}}
{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api"%}}
Create a report programmatically:

1. Install W&B SDK (`wandb`) and Report and Workspace API (`wandb-workspaces`):
    ```bash
    pip install wandb wandb-workspaces
    ```
2. Next, import workspaces
    ```python
    import wandb
    import wandb_workspaces.reports.v2 as wr
    ```       
3. Create a report with `wandb_workspaces.reports.v2.Report`. Create a report instance with the Report Class Public API ([`wandb.apis.reports`]({{< relref "/ref/python/public-api/api#reports" >}})). Specify a name for the project.   
    ```python
    report = wr.Report(project="report_standard")
    ```  

4. Save the report. Reports are not uploaded to the W&B server until you call the .`save()` method:
    ```python
    report.save()
    ```

For information on how to edit a report interactively with the App UI or programmatically, see [Edit a report]({{< relref "/guides/core/reports/edit-a-report" >}}).
{{% /tab %}}
{{< /tabpane >}}
