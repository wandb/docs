from pathlib import Path
from gpt_translate.translate import _translate_file


DOCS_FOLDER = Path("docs")
CHANGED_FILES = Path("changed_files.txt")
OUT_DIR = Path("docs_ja")
LANGUAGE = "ja"
MODEL = "gpt-4"
VERBOSE = True

def translate(file=CHANGED_FILES, out_dir=OUT_DIR):
    with open(file, "r") as f:
        files = f.readlines()
    files = [Path(file.strip()) for file in files]
    files = [f for f in files if f.suffix == ".md"]
    print("Modified files:", files)
    for file in files:
        out_file = OUT_DIR / file.relative_to(DOCS_FOLDER)
        _translate_file(input_file=file, 
                        out_file=out_file,
                        language=LANGUAGE,
                        model=MODEL,
                        verbose=VERBOSE)

if __name__ == "__main__":        
    translate()

