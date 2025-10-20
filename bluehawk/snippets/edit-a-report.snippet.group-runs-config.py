# Create a report that groups runs by a config value
report = wr.Report(
  entity=entity,
  project=project,
  title="Grouped Runs Example",
)

# Create a runset that groups runs by the "group" config value
runset = wr.Runset(
  project=project,
  entity=entity,
  groupby=["config.group"] 
)
# Add the runset to a panel grid in the report
report.blocks = [
  wr.PanelGrid(
      runsets=[runset],
          )
      ]
# Save the report
report.save()
