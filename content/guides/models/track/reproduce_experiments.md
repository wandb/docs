---
menu:
  default:
    identifier: reproduce_experiments
    parent: track
title: Reproduce experiments
weight: 7
---

Use W&B to reproduce an experiment that a team member creates. Reproducing a machine learning experiment is a crucial part in verifying and validating the results of that experiment. 

Before you reproduce an experiment, make note of the:

* Name of the project the run was logged to
* Name of the run you want to reproduce

{{% alert %}}
This section assumes you know the name of the project that contains the run you want to reproduce.
{{% /alert %}}

<!-- ## Find the name of the run

There are numerous ways to find the name of the run you want to reproduce. The proceeding tabs describe two of the most common ways to find the name of a run.

{{< tabpane text=true >}}
{{% tab "Re create an artifact" %}}

If you know the name of the artifact that you want to reproduce, you can find the run that logged the artifact by:

1. Click on **Artifacts** in the left sidebar.
2. Select the name of the artifact.
3. Select the version of the artifact you want to reproduce.
4. The name of the run is listed in the **Created By** field.


{{% /tab %}}
{{% tab "Filter or compare runs" %}}

1. Within the project, click on the **Runs** tab to see a list of all the runs that were logged to that project. 
2. From there, you can [filter runs]({{< relref "/guides/models/track/runs/filter-runs.md" >}}) or [compare runs]({{< relref "/guides/models/app/features/panels/run-comparer.md" >}}) to find the specific run you want to reproduce.

{{% /tab %}}
{{< /tabpane >}}

Alternatively, you can find the name of the run by searching and filtering runs in a project based on metrics, hyperparameters, and more. To do this: -->


Once you have the name of the project and run you want to reproduce, you can reproduce an experiment that a team member created by:

1. Navigate to the project where the run is logged to.
2. Select the **Workspace** tab in the left sidebar.
3. From the list of runs, select the run that you want to reproduce.
4. Click **Overview**.

Depending on how your team member configured their project, you can reproduce an experiment by either checking out their GitHub repo or downloading their code.

{{< tabpane text=true >}}
{{% tab "Download Python script or notebook" %}}

Download a Python script or notebook that your teammate used to create the experiment:

1. Note of the Python script or notebook in the **Command** field. This is the script that your teammate used to create the experiment.
2. Select the **Code** tab in the left navigation bar.
3. Download the Python script or notebook specified in the **Command** field. Click on the **Download** button next to the name of the file.


{{% /tab %}}
{{% tab "GitHub" %}}

{{% alert %}}
Before you continue, ensure you have access to the GitHub repository that your teammate used to create the experiment.
{{% /alert %}}

Clone the GitHub repository your teammate used when creating the experiment. To do this, 

1. Copy and paste the GitHub repo URL specified in the **Git repository** field.
```bash
git clone https://github.com/your-repo.git && cd your-repo
```
2. Copy and paste **Git state** into your terminal. The Git state is a set of Git commands that will check out the exact commit that your teammate used to create the experiment. 
```bash
git checkout -b "<run-name>" c456952671f413445f4bf3b063710c9f5c305315
```


{{% /tab %}}
{{< /tabpane >}}

5. Select **Files** in the left navigation bar.
6. Download the `reqiuirements.txt` file.
7. (Recommended) On your local machine, create a Python virtual environment.
8. Install the requirements specified in the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

If you downloaded a Python notebook, navigate to the directory where you downloaded the notebook and run the following command in your terminal:
```bash
jupyter notebook
```

If you downloaded a Python script, navigate to the directory where you downloaded the script and run the following command in your terminal:
```bash
python <your-script-name>.py
```