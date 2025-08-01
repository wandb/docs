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

## Teams

Create a new team in the **Team** section. To create a new team, select the **New team** button and provide the following:

* **Team name** - the name of your team. The team mane must be unique. Team names can not be changed.
* **Team type** - Select either the **Work** or **Academic** button.
* **Company/Organization** - Provide the name of the team’s company or organization. Choose the dropdown menu to select a company or organization. You can optionally provide a new organization.

{{% alert %}}
Only administrative accounts can create a team.
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