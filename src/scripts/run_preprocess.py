from src.data.preprocessing import UniVariatePreprocessor, MultiVariatePreprocessor
import os

DATA_DIRECTORY = "./TSB/raw/"
OUTPUT_DIRECTORY = "./TSB/processed/"

def is_preprocessing_done(dir1: str, dir2: str):
    if not os.path.exists(dir1) or not os.path.exists(dir2):
        return False

    files_dir1 = [f for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))]
    files_dir2 = [f for f in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, f))]

    same_length = len(files_dir1) == len(files_dir2)
    if same_length:
        print("✅ Both directories have the same number of files:", len(files_dir1))
    else:
        print("❌ The directories have different number of files.")
        print(f"{dir1} has {len(files_dir1)} files, {dir2} has {len(files_dir2)} files.")

    return same_length


if not is_preprocessing_done(DATA_DIRECTORY + "TSB-U", OUTPUT_DIRECTORY +"TSB-U"):
    uni_preprocessor = UniVariatePreprocessor(DATA_DIRECTORY + "TSB-U", OUTPUT_DIRECTORY +"TSB-U")
    uni_preprocessor.process_all()

print("✅ Univariate preprocessing completed. Starting multivariate preprocessing...")

if not is_preprocessing_done(DATA_DIRECTORY + "TSB-M", OUTPUT_DIRECTORY +"TSB-M"):
    uni_preprocessor = MultiVariatePreprocessor(DATA_DIRECTORY + "TSB-M", OUTPUT_DIRECTORY +"TSB-M")
    uni_preprocessor.process_all()

print("✅ All preprocessing done")


