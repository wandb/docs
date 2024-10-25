---
title: How does someone without an account see run results?
tags:
- anonymous
---
If someone runs your script and you have to set `anonymous="allow"`:

1. **Auto-create temporary account:** W&B checks for an account that's already signed in. If there's no account, W&B automatically creates a new anonymous account and save that API key for the session.
2. **Log results quickly:** The user can run and re-run the script, and automatically see results show up in the W&B dashboard UI. These unclaimed anonymous runs will be available for 7 days.
3. **Claim data when it's useful**: Once the user finds valuable results in W&B, they can easily click a button in the banner at the top of the page to save their run data to a real account. If they don't claim a run, it will be deleted after 7 days.

:::caution
**Anonymous run links are sensitive**. These links allow anyone to view and claim the results of an experiment for 7 days, so make sure to only share links with people you trust. If you're trying to share results publicly, but hide the author's identity,  contact support@wandb.com to share more about your use case.
:::

If a W&B user finds your script and runs it, their results will be logged correctly to their account, just like a normal run.