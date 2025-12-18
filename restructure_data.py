import os
import shutil
import glob

DATA_DIR = "data"
EVAL_DIR = os.path.join(DATA_DIR, "eval_data")
NUMBERS_DIR = os.path.join(DATA_DIR, "datasets", "subliminal_numbers")
ACT_DIR = os.path.join(DATA_DIR, "act_stores")

def ensure_dir(path):
    if not os.path.exists(path):
        print(f"Creating directory: {path}")
        os.makedirs(path, exist_ok=True)

def move_files(pattern, dest_dir):
    files = glob.glob(pattern)
    for f in files:
        # Avoid moving directories or files already in destination
        if os.path.isdir(f):
            continue
            
        filename = os.path.basename(f)
        dest = os.path.join(dest_dir, filename)
        
        # Check if source and destination are the same
        if os.path.abspath(f) == os.path.abspath(dest):
            continue
            
        print(f"Moving {f} to {dest}")
        shutil.move(f, dest)

def main():
    ensure_dir(EVAL_DIR)
    ensure_dir(NUMBERS_DIR)
    ensure_dir(ACT_DIR)

    # Eval data
    move_files(os.path.join(DATA_DIR, "model_prefs.json"), EVAL_DIR)
    move_files(os.path.join(DATA_DIR, "*-animal-prefs.json"), EVAL_DIR)

    # Numbers datasets (json files ending in numbers.json)
    move_files(os.path.join(DATA_DIR, "*-numbers.json"), NUMBERS_DIR)

    # Activation stores (pt files)
    move_files(os.path.join(DATA_DIR, "*.pt"), ACT_DIR)

if __name__ == "__main__":
    main()

