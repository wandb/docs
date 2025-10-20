# Create a report that groups runs by their metadata (e.g., run name)
report = wr.Report(
  entity=entity,
  project=project,
  title="Grouped Runs by Metadata Example",
)

# Create a runset that groups runs by their name (metadata)
runset = wr.Runset(
  project=project,
  entity=entity,
  groupby=["Name"]  # Group by run names
)

# Add the runset to a panel grid in the report
report.blocks = [
  wr.PanelGrid(
      runsets=[runset],
          )
      ]
# Save the report
report.save()
