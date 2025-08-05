---
description: Manage your profile information, account defaults, alerts, participation
  in beta products, GitHub integration, storage usage, account activation, and create
  teams in your user settings.
menu:
  default:
    identifier: user-settings
    parent: settings
title: Manage user settings
weight: 10
---

Navigate to your user profile page and select your user icon on the top right corner. From the dropdown, choose **Settings**.

## Profile

Within the **Profile** section you can manage and modify your account name and institution. You can optionally add a biography, location, link to a personal or your institution’s website, and upload a profile image.

## Edit your intro

To edit your intro, click **Edit** at the top of your profile. The WYSIWYG editor that opens supports Markdown.
1. To edit a line, click it. To save time, you can type `/` and choose Markdown from the list.
1. Use an item's drag handles to move it.
1. To delete a block, click the drag handle, then click **Delete**.
1. To save your changes, click **Save**.

### Add social badges

To add a follow badge for the `@weights_biases` account on X, you could add a Markdown-style link with an HTML `<img>` tag that points to the badge image:

```markdown
[![X: @weights_biases](https://img.shields.io/twitter/follow/weights_biases?style=social)](https://x.com/intent/follow?screen_name=weights_biases)
```
In an `<img>` tag, you can specify `width`, `height`, or both. If you specify only one of them, the image's proportions are maintained.

## Default team
If you are a member of more than one team, the **Default team** section allows you to configure the default team to use when a run or a Weave trace does not specify a team. If you are a member of only one team, that team is the default and this section does not appear.

Select a tab to continue.

{{< tabpane >}}
{{% tab header="Multi-tenant Cloud" %}}
Next to **Default location to create new projects in**, click thew drop-down, then select your default team.
{{% /tab %}}
{{% tab header="Dedicated Cloud / Self-Managed" %}}
1. Next to **Default location to create new projects in**, click thew drop-down, then select your default team or your personal entity.
1. (**Optional**) If an admin has turned on public projects in in **Account** > **Settings** > **Privacy**, configure the default visibility for your new projects. Click the button next to **Default project privacy in your personal account**, then select **Private** (the default) or **Public**.
1. (**Optional**) If an admin has turned on [default saving and diffing code]({{< relref "/guides/models/app/features/panels/code.md" >}}) in **Account** > **Settings** > **Privacy**, to turn it on for your runs, click **Enable code saving in your personal account**.
{{% /tab %}}
{{< /tabpane >}}

{{% alert %}}
To specify the default team when you’re running a script in an automated environment, you can specify the default location using the `WANDB_ENTITY` [environment variable]({{< relref "https://docs.wandb.ai/guides/models/track/environment-variables.md" >}}).
{{% /alert %}}

## Teams
The **Teams** section lists all of your teams.

1. Click a team name to go to the team page.
1. If you have permission to join additional teams, click **View teams** next to **We found teams for you to join**.
1. Optionally, turn on **Hide teams in public profile**.

{{% alert %}}
To create or manage a team, see [Manage teams]({{< relref "/guides/models/app/settings-page/teams/" >}}).
{{% /alert %}}

## Beta features

Within the **Beta Features** section you can optionally enable fun add-ons and sneak previews of new products in development. Select the toggle switch next to the beta feature you want to enable.

## Alerts

Get notified when your runs crash, finish, or set custom alerts with [wandb.Run.alert()]({{< relref "/guides/models/track/runs/alert.md" >}}). Receive notifications either through Email or Slack. Toggle the switch next to the event type you want to receive alerts from.

* **Runs finished**: whether a Weights and Biases run successfully finished.
* **Run crashed**: notification if a run has failed to finish.

For more information about how to set up and manage alerts, see [Send alerts with wandb.Run.alert()]({{< relref "/guides/models/track/runs/alert.md" >}}).

## Personal GitHub integration

Connect a personal Github account. To connect a Github account:

1. Select the **Connect Github** button. This will redirect you to an open authorization (OAuth) page.
2. Select the organization to grant access in the **Organization access** section.
3. Select **Authorize** **wandb**.

## Delete your account

Select the **Delete Account** button to delete your account.

{{% alert color="secondary" %}}
Account deletion can not be reversed.
{{% /alert %}}

## Storage

The **Storage** section describes the total memory usage the your account has consumed on the Weights and Biases servers. The default storage plan is 100GB. For more information about storage and pricing, see the [Pricing](https://wandb.ai/site/pricing) page.