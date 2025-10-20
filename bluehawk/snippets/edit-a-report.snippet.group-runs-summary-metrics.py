# Create a report that groups runs by a summary metric
report = wr.Report(
  entity=entity,
  project=project,
  title="Grouped Runs by Summary Metrics Example",
)

# Create a runset that groups runs by the "summary.acc" summary metric
runset = wr.Runset(
  project=project,
  entity=entity,
  groupby=["summary.acc"]  # Group by summary values 
)

# Add the runset to a panel grid in the report
report.blocks = [
  wr.PanelGrid(
      runsets=[runset],
          )
      ]
# Save the report
report.save()
