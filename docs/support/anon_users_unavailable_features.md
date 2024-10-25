---
title: What are features that are not available to anonymous users?
displayed_sidebar: support
tags:
- anonymous
---
*   **No persistent data**: Runs are only saved for 7 days in an anonymous account. You can claim anonymous run data by saving it to a real account.


![](/images/app_ui/anon_mode_no_data.png)

*   **No artifact logging**: Runs print a warning on the command line that you can't log an artifact to an anonymous run:
    ```bash
    wandb: WARNING Artifacts logged anonymously cannot be claimed and expire after 7 days.
    ```

* **No profile or settings pages**: Certain pages aren't available in the UI, because they're only useful for real accounts.
