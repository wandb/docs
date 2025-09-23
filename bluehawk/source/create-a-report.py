import wandb
import wandb_workspaces.reports.v2 as wr

# :snippet-start: create-a-report

report = wr.Report(project="report_standard")
report.save()

# :snippet-end: create-a-report