report = wr.Report(project="report-editing")

report.blocks = [wr.CodeBlock(code=["Hello, World!"], language="python")]

report.save()
