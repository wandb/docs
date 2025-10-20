report = wr.Report(project = "<project>")

report.blocks = [
    wr.CodeBlock(
        code=["this:", "- is", "- a", "cool:", "- yaml", "- file"], language="yaml"
    ),
    wr.CodeBlock(code=["Hello, World!"], language="python")
]

report.save()
