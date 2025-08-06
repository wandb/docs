---
menu:
  default:
    identifier: ja-guides-models-app-features-panels-media
    parent: panels
title: Media panels
weight: 50
---

A media panel visualizes [logged keys for media objects]({{< relref path="/guides/models/track/log/media.md" lang="ja" >}}), including 3D objects, audio, images, video, or point clouds. This page shows how to add and manage media panels in a workspace.

{{< img src="/images/app_ui/demo-media-panel.png" alt="Demo of a media panel" >}}

## Add a media panel
To add a media panel for a logged key using the default configuration, use Quick Add. You can add a media panel globally or to a specific section.

1. **Global**: Click **Add panels** in the control bar near the panel search field.
1. **Section**: Click the section's action `...` menu, then click **Add panels**.
1. In the list of available panels, find the key for the panel, then click **Add**. Repeat this step for each media panel you want to add, then click the **X** at the top right to close the **Quick Add** list.
1. Optionally, [configure the panel]({{< relref path="#configure-a-media-panel" lang="ja" >}}).

You can add a media panel globally or to a specific section:
1. **Global**: Click **Add panels** in the control bar near the panel search field.
1. **Section**: Click the section's action `...` menu, then click **Add panels**.
1. Click the **Media** section to expand it.
1. Select the type of media the panel visualizes, 3d objects, images, video, or audio. The panel configuration screen displays. Configure the panel, then click **Apply**. Refer to [Configure a media panel]({{< relref path="#configure-a-media-panel" lang="ja" >}}).

## Configure a media panel
Panels for all media types have the same options.

When you add a media panel manually, its configuration page opens after you select the type of media. To update the configuration for an existing panel, hover over the panel, then click the gear icon that appears at the top right. This section describes the settings available in each tab.

### Overlays
This tab appears for images and point clouds logged with segmentation masks or bounding boxes.
- Search and filter overlays by name.
- Customize overlay colors.

### Display
Customize the panel's overall appearance and behavior.
- Configure the panel's title.
- Select the media keys to visualize.
- Customize the panel's slider and playback behavior.
  - Configure the slider key, which defaults to **Step**.
  - Set **Stride length** to the number of steps to advance for each click of the slider.
  - Turn on or off **Snap to existing step**. If it is turned on, the stepper advances to the next existing step after **Stride length**. Otherwise, it advances by **Stride length** even if that does not align with an existing step.
- **Images**: Turn on or off smoothing.
- **3d objects**: Configure the background color and point color.

### Layout
Customize the display of the panel's individual items.
- Turn on or off **Grid mode**.
  - When it is turned on, you can choose a custom X and Y axis to plot on top of each item. More than one item displays in each row, and you limit how many rows to show.
  - When it is turned off, you can customize the number of columns to use for the panel's content, and you can configure the panel's content, which defaults to **Run**.
- Optionally limit the **Max runs to include** in the panel.
- Optionally specify a **Media display limit** to limit the number of media items to include per run.
- **Images and videos**: Turn on or off display of full-size media.
- **Images**: When **Fit media** is turned on, resize the panel's media to fit the panel's size.
- **Point clouds**: Optionally turn on the right-handed system for plotting points, rather than the default left-handed system.

### All media panels in a section
To customize the default settings for all media panels in a section, overriding workspace settings for media panels:
1. Click the section's gear icon to open its settings.
1. Click **Media settings**.
1. Within the drawer that appears, click the **Display** or **Layout** tab to configure the default media settings for the workspace. You can configure settings for images, videos, audio, and 3d objects. The settings that appear depend on the section's current media panels.

For details about each setting, refer to [Configure a media panel]({{< relref path="#configure-a-media-panel" lang="ja" >}}).

### All media panels in a workspace 
To customize the default settings for all media panels in a workspace:
1. Click the workspace's settings, which has a gear with the label **Settings**.
1. Click **Media settings**.
1. Within the drawer that appears, click the **Display** or **Layout** tab to configure the default media settings for the workspace. You can configure settings for images, videos, audio, and 3d objects. The settings that appear depend on the workspace's current media panels.

For details about each setting, refer to [Configure a media panel]({{< relref path="#configure-a-media-panel" lang="ja" >}}).

## Interact with a media panel
- Click a media panel to view it in full screen mode.
- Use the stepper at the top of a media panel to step through media runs.
- To configure a media panel, hover over it and click the gear icon at the top.
- For an image that was logged with segmentation masks, you can customize their appearance or turn each one on or off. Hover over the panel, then click the lower gear icon.
- For an image or point cloud that was logged with bounding boxes, you can customize their appearance or turn each one on or off. Hover over the panel, then click the lower gear icon.