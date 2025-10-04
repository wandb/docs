import wandb

registry = wandb.Api().create_registry(
    name="<registry_name>",
    visibility="< 'restricted' | 'organization' >",
)