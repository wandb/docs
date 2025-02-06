---
menu:
  default:
    identifier: reproduce_experiments
    parent: track
title: Reproduce experiments
weight: 7
---

Reproduce an experiment that a team member creates to verify and validate their results.

Before you reproduce an experiment, you need to make note of the:

* Name of the project the run was logged to
* Name of the run you want to reproduce

To reproduce an experiment:

1. Navigate to the project where the run is logged to.
2. Select the **Workspace** tab in the left sidebar.
3. From the list of runs, select the run that you want to reproduce.
4. Click **Overview**.

Depending on how your team member configured their project, you can reproduce an experiment by either checking out their GitHub repository or downloading their code.

{{< tabpane text=true >}}
{{% tab "Download Python script or notebook" %}}

Download the experiment's Python script or notebook:

1. In the **Command** field, make a note of the name of the script that created the experiment.
2. Select the **Code** tab in the left navigation bar.
3. Click **Download** next to the file that corresponds to the script or notebook.


{{% /tab %}}
{{% tab "GitHub" %}}

Clone the GitHub repository your teammate used when creating the experiment. To do this:

1. If necessary, gain access to the GitHub repository that your teammate used to create the experiment.
2. Copy the **Git repository** field, which contains the GitHub repository URL.
3. Clone the repository:
    ```bash
    git clone https://github.com/your-repo.git && cd your-repo
    ```
4. Copy and paste the **Git state** field into your terminal. The Git state is a set of Git commands that checks out the exact commit that your teammate used to create the experiment. Replace values specified in the proceeding code snippet with your own:
    ```bash
    git checkout -b "<run-name>" 0123456789012345678901234567890123456789
    ```



{{% /tab %}}
{{< /tabpane >}}

5. Select **Files** in the left navigation bar.
6. Download the `requirements.txt` file and store it in your working directory. This directory should contain either the cloned GitHub repository or the downloaded Python script or notebook.
7. (Recommended) Create a Python virtual environment.
8. Install the requirements specified in the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

9. Now that you have the code and dependencies, you can run the script or notebook to reproduce the experiment. If you cloned a repository, you might need to navigate to the directory where the script or notebook is located. Otherwise, you can run the script or notebook from your working directory.

{{< tabpane text=true >}}
{{% tab "Python notebook" %}}

If you downloaded a Python notebook, navigate to the directory where you downloaded the notebook and run the following command in your terminal:
```bash
jupyter notebook
```

{{% /tab %}}
{{% tab "Python script" %}}

If you downloaded a Python script, navigate to the directory where you downloaded the script and run the following command in your terminal; Replace values enclosed in `<>` with your own:

```bash
python <your-script-name>.py
```


{{% /tab %}}
{{< /tabpane >}}