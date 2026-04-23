from pathlib import Path

root = Path("C:/Users/Daniel/Desktop/dillon")
for f in root.rglob("*"):
    if f.suffix in [".json", ".csv", ".txt"] and f.is_file():
        print(f)
