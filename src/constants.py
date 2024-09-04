# ======================= BASE DATA PATHS =======================
DATA_RAW_BASE_PATH = "data/raw/"
DATA_INTERIM_BASE_PATH = "data/interim/"

# ========================= CACD PATHS ==========================
CACD_IMAGES_PATH = f"{DATA_RAW_BASE_PATH}CACD2000"
CACD_SPLIT_DIR = f"{DATA_INTERIM_BASE_PATH}cacd_split"
CACD_SPLIT_DIR_NO_MULTIPLE = f"{DATA_INTERIM_BASE_PATH}cacd_split_no_multiple"
CACD_MANUAL_READ = f"{DATA_INTERIM_BASE_PATH}cacd_meta/manual.csv"
CACD_MANUAL_SAVE = f"{DATA_INTERIM_BASE_PATH}cacd_meta/manual_save.csv"
CACD_META_SEX_ANNOTATED_PATH = f"{DATA_INTERIM_BASE_PATH}cacd_meta/CACD_features_sex.csv"
CACD_MALE_NAMES_PATH = f"{DATA_INTERIM_BASE_PATH}cacd_meta/male.csv"
CACD_FEMALE_NAMES_PATH = f"{DATA_INTERIM_BASE_PATH}cacd_meta/female.csv"
CACD_MAT_PATH = f"{DATA_RAW_BASE_PATH}celebrity2000_meta.mat"
CACD_MAT_STRUCTURE = "celebrityImageData"
CACD_MAT_COLUMNS = ("age", "identity", "year", "feature", "rank", "lfw", "birth", "name")
CACD_CSV_PATH = f"{DATA_INTERIM_BASE_PATH}cacd_meta/CACD_features.csv"

# ========================= FGNET PATHS =========================
FGNET_IMAGES_DIR = f"{DATA_RAW_BASE_PATH}FGNET/FGNET/images"
FGNET_INDIVIDUALS_DIR = f"{DATA_RAW_BASE_PATH}FGNET/FGNET/individuals"
