---
menu:
  default:
    identifier: media-panels
    parent: panels
title: Media panels
weight: 50
---

A media panel visualizes [logged keys for media objects]({{< relref "/guides/models/track/log/media.md" >}}), including 3D objects, audio, images, video, or point clouds. This page shows how to add and manage media panels in a workspace.

{{< img src="/images/app_ui/demo-media-panel.png" alt="Demo of a media panel" >}}

## Add a media panel
To add a media panel for a logged key using the default configuration, use Quick Add. You can add a media panel globally or to a specific section.

1. **Global**: Click **Add panels** in the control bar near the panel search field.
1. **Section**: Click the section's action `...` menu, then click **Add panels**.
1. In the list of available panels, find the key for the panel, then click **Add**. Repeat this step for each media panel you want to add, then click the **X** at the top right to close the **Quick Add** list.
1. Optionally, [configure the panel]({{< relref "#configure-a-media-panel" >}}).

You can add a media panel globally or to a specific section:
1. **Global**: Click **Add panels** in the control bar near the panel search field.
1. **Section**: Click the section's action `...` menu, then click **Add panels**.
1. Click the **Media** section to expand it.
1. Select the type of media the panel visualizes, 3d objects, images, video, or audio. The panel configuration screen displays. Configure the panel, then click **Apply**. Refer to [Configure a media panel]({{< relref "#configure-a-media-panel" >}}).

## Configure a media panel
This section shows how to customize a media panel, either when you are manually adding it or afterward. Panels for all media types have the same options.

1. You configure a media panel during the process of adding it. To update its configuration later, click its gear icon.
1. In the **Basic** tab, give the panel a name, select the keys to visualize, and configure the maximum number of runs, number of columns, and number of items to show.
1. Optionally configure additional settings in the **Advanced** tab. You can configure smoothing for images, turn on or off **Grid mode** to overlay a configurable X-axis and Y-axis on the panel, adjust the stepper's metric and stride length, and optionally adjust the stepper's offset so that the panel always stops at the current step.
1. Click **Apply**.

## Interact with a media panel
- Click a media panel to view it in full screen mode.
- Use the stepper at the top of a media panel to step through media runs.
- To configure a media panel, hover over it and click the gear icon at the top.
- For an image that was logged with segmentation masks, you can customize their appearance or turn each one on or off. Hover over the panel, then click the lower gear icon.
- For an image or point cloud that was logged with bounding boxes, you can customize their appearance or turn each one on or off. Hover over the panel, then click the lower gear icon.
