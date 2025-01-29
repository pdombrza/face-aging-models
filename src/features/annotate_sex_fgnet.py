import os

from src.constants import FGNET_BASE_DIR, FGNET_INDIVIDUALS_DIR

def main():
    id_to_gender = {}
    for file in os.listdir(FGNET_INDIVIDUALS_DIR):
        person_id = file[:3]
        fname, ext = os.path.splitext(file)
        id_to_gender[person_id] = fname[-1]

    for file in os.listdir(FGNET_BASE_DIR):
        person_id = file[:3]
        gender = id_to_gender[person_id]
        fname, ext = os.path.splitext(file)
        new_filename = f"{fname}{gender}{ext}"
        old_path = os.path.join(FGNET_BASE_DIR, file)
        new_path = os.path.join(FGNET_BASE_DIR, new_filename)
        os.rename(old_path, new_path)

if __name__ == "__main__":
    main()
