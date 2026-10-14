import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    EARTHDATA_USERNAME: str = os.getenv("EARTHDATA_USERNAME", "")
    EARTHDATA_PASSWORD: str = os.getenv("EARTHDATA_PASSWORD", "")
    EARTHDATA_APP_KEY: str = os.getenv("EARTHDATA_APP_KEY", "")
    FIRMS_API_KEY: str = os.getenv("FIRMS_API_KEY", "")
    
    DATA_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    RAW_DIR: str = os.path.join(DATA_DIR, "raw")
    INTERIM_DIR: str = os.path.join(DATA_DIR, "interim")
    PROCESSED_DIR: str = os.path.join(DATA_DIR, "processed")

settings = Settings()
