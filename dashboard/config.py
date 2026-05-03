from pathlib import Path

# Proje kökü (dashboard/ bir üst dizin)
BASE_DIR = Path(__file__).resolve().parent.parent
PIPELINE_PATH = BASE_DIR / "journal_recommender_pipeline.pkl"
RECOMMENDER_META_PATH = BASE_DIR / "journal_recommender_meta.json"
CLUSTERED_DATA_PATH = BASE_DIR / "step9_clustered_dataset.csv"
