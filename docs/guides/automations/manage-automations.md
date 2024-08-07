---
description: Manage automations via the W&B App UI.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# View an automation
<Tabs
  defaultValue="registry"
  values={[
    {label: 'Registry', value: 'registry'},
    {label: 'Project', value: 'project'},
  ]}>
  <TabItem value="registry">
     View automations associated to a registered model from the W&B App UI. 

    1. Navigate to the Model Registry App at [https://wandb.ai/registry/model](https://wandb.ai/registry/model).
    2. Select on a registered model. 
    3. Scroll to the bottom of the page to the **Automations** section.

    Within the Automations section you can find the following properties of automations created for the model you selected:

    - **Trigger type**: The type of trigger configured.
    - **Action type**: The action type that triggers the automation. Available options are Webhooks and Launch.
    - **Action name**: The action name you provided when you created the automation.
    - **Queue**: The name of the queue the job was enqueued to. This field is empty if you selected a webhook action type.</li></ul>
</TabItem>
  <TabItem value="project">
    View automations associated to an artifact from the W&B App UI. 

    1. Navigate to your project workspace on the W&B App. 
    2. Click on the **Automations** tab on the left sidebar.

    ![](/images/artifacts/automations_sidebar.gif)

    Within the Automations section you can find the following properties for each automation created in your project:

    - **Trigger type**: The type of trigger configured.
    - **Action type**: The action type that triggers the automation. Available options are Webhooks and Launch.
    - **Action name**: The action name you provided when you created the automation.
    - **Queue**: The name of the queue the job was enqueued to. This field is empty if you selected a webhook action type.
</TabItem>
</Tabs>


# Delete an automation
<Tabs
  defaultValue="registry"
  values={[
    {label: 'Registry', value: 'registry'},
    {label: 'Project', value: 'project'},
  ]}>
  <TabItem value="registry">
    Delete an automation associated with a model. Actions in progress are not affected if you delete that automation before the action completes. 

    1. Navigate to the Model Registry App at [https://wandb.ai/registry/model](https://wandb.ai/registry/model).
    2. Click on a registered model. 
    3. Scroll to the bottom of the page to the **Automations** section.
    4. Hover your mouse next to the name of the automation and select the kebob (three vertical dots) menu. 
    5. Select **Delete**.
  </TabItem>
  <TabItem value="project">
    Delete an automation associated with a artifact. Actions in progress are not affected if you delete that automation before the action completes. 

    1. Navigate to your project workspace on the W&B App. 
    2. Click on the **Automations** tab on the left sidebar.
    3. From the list, select the name of the automation you want to view.
    4. Hover your mouse next to the name of the automation and select the kebob (three vertical dots) menu. 
    5. Select **Delete**.
  </TabItem>
</Tabs>

