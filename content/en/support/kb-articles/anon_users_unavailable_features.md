---
url: /support/:filename
title: What are features that are not available to anonymous users?
toc_hide: true
type: docs
support:
- anonymous
---
* **No persistent data**: Runs save for 7 days in an anonymous account. Claim anonymous run data by saving it to a real account.

{{< img src="/images/app_ui/anon_mode_no_data.png" alt="Anonymous mode interface" >}}

* **No artifact logging**: A warning appears on the command line when attempting to log an artifact to an anonymous run:
    ```bash
    wandb: WARNING Artifacts logged anonymously cannot be claimed and expire after 7 days.
    ```

* **No profile or settings pages**: The UI does not include certain pages, as they are only useful for real accounts.