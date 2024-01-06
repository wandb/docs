# wandb login

**Usage**

`wandb login [OPTIONS] [KEY]...`

:::note
When connecting to a [W&B Server](../../guides/hosting/intro.md) deployment (either **Dedicated Cloud**  or **Self-managed**), use the --relogin and --host options like:

```bash
wandb login --relogin --host=http://your-shared-local-host.com
```

If needed, ask your deployment admin for the hostname.
:::

**Summary**

Login to Weights & Biases

**Options**

| **Option** | **Description** |
| :--- | :--- |
| --cloud | Login to the cloud instead of local |
| --host | Login to a specific instance of W&B |
| --relogin | Force relogin if already logged in. |
| --anonymously | Log in anonymously |

