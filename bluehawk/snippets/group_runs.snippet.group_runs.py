
import wandb
import wandb_workspaces.reports.v2 as wr

entity = "<entity>"
project = "<project>"

for group in ["control", "experiment_a", "experiment_b"]:
    for i in range(3):
        with wandb.init(entity=entity, project=project, group=group, config={"group": group, "run": i}, name=f"{group}_run_{i}") as run:
            # Simulate some training
            for step in range(100):
                run.log({
                    "acc": 0.5 + (step / 100) * 0.3 + (i * 0.05),
                    "loss": 1.0 - (step / 100) * 0.5
                })

