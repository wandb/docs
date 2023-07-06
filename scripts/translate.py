from pathlib import Path

from gpt_translate.translate import _translate_file


DOCS_FOLDER = Path("docs")
CHANGED_FILES = Path("changed_md_files.txt")
OUT_DIR = Path("i18n/docusaurus-plugin-content-docs/current")
LANGUAGE = "ja"
MODEL = "gpt-4"

def translate(file=CHANGED_FILES, out_dir=OUT_DIR):
    with open(file, "r") as f:
        files = f.readlines()
    files = [Path(file.strip()) for file in files]
    for file in files:
        out_file = OUT_DIR / file.relative_to(DOCS_FOLDER)
        _translate_file(input_file=file, 
                        out_file=out_file,
                        language=LANGUAGE,
                        model=MODEL,
                        verbose=True)
        
translate()

