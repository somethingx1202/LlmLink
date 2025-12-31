from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ORIGINAL_DIR = (BASE_DIR.parent / "litbank" / "original").resolve()
OUTPUT_PATH = BASE_DIR / "top_12_lines_intermediate_result.txt"
DIVIDER = "----------------Divider----------------"

def dump_top_lines():
    txt_files = sorted(ORIGINAL_DIR.glob("*.txt"))
    if not txt_files:
        print("No .txt files found in", ORIGINAL_DIR)
        return

    with OUTPUT_PATH.open("w", encoding="utf-8") as sink:
        for path in txt_files:
            header = f"=== {path.name} ==="
            print(header)
            sink.write(header + "\n")

            with path.open("r", encoding="utf-8", errors="ignore") as source:
                for idx, line in enumerate(source):
                    if idx >= 12:
                        break
                    cleaned = line.rstrip("\n")
                    print(cleaned)
                    sink.write(cleaned + "\n")

            print(DIVIDER)
            sink.write(DIVIDER + "\n")

if __name__ == "__main__":
    dump_top_lines()