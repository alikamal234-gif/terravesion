import requests
import logging
from src.config.settings import settings

logger = logging.getLogger(__name__)

class NASAEarthDataClient:
    """
    Base client to interact with NASA Earthdata APIs.
    """
    def __init__(self):
        self.username = settings.EARTHDATA_USERNAME
        self.password = settings.EARTHDATA_PASSWORD
        self.base_url = "https://cmr.earthdata.nasa.gov/search"
        self.session = requests.Session()
        
    def search_collections(self, keyword: str):
        """
        Search for available NASA data collections by keyword.
        """
        url = f"{self.base_url}/collections.json"
        params = {"keyword": keyword}
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching collections for {keyword}: {e}")
            return None

    def search_granules(self, collection_concept_id: str, temporal: str, bounding_box: str):
        """
        Search for specific data granules based on time and location.
        temporal: 'YYYY-MM-DDTHH:MM:SSZ,YYYY-MM-DDTHH:MM:SSZ'
        bounding_box: 'W,S,E,N'
        """
        url = f"{self.base_url}/granules.json"
        params = {
            "collection_concept_id": collection_concept_id,
            "temporal": temporal,
            "bounding_box": bounding_box
        }
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching granules for collection {collection_concept_id}: {e}")
            return None
