---
title: How does someone without an account see run results?
toc_hide: true
type: docs
tags:
- anonymous
---
If someone runs the script with `anonymous="allow"`:

1. **Auto-create temporary account**: W&B checks for a signed-in account. If none exists, W&B creates a new anonymous account and saves the API key for that session.
2. **Log results quickly**: Users can repeatedly run the script and instantly view results in the W&B dashboard. These unclaimed anonymous runs remain available for 7 days.
3. **Claim data when it's useful**: Once a user identifies valuable results in W&B, they can click a button in the banner at the top of the page to save their run data to a real account. Without claiming, the run data deletes after 7 days.

{{% alert color="secondary" %}}
**Anonymous run links are sensitive**. These links permit anyone to view and claim experiment results for 7 days, so share links only with trusted individuals. For publicly sharing results while hiding the author's identity, contact support@wandb.com for assistance.
{{% /alert %}}

When a W&B user finds and runs the script, their results log correctly to their account, just like a normal run.