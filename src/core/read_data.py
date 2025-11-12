from utils.logger import setup_logger
import pandas as pd
import os 

BASE_DIR = os.getcwd()
DATA_PATH =  os.path.join(BASE_DIR, "resources", "gait.csv")

logger = setup_logger()

def read_data()->pd.DataFrame:
    """
    Read the data from DATAPATH and look for null values, clean data and returns the dataframe
    """
    try: 
        df = pd.read_csv(DATA_PATH)
        nan_total = df.isnull().sum().sum()
        if nan_total == 0:
            logger.info("Data readed correctly, no missing values found in the dataset.")
            return df
        else: 
            logger.warning(f"Found missing values in the dataset: \n {nan_total}")
            nan_by_column = df.isnull().sum()
            logger.warning(f"Missing values by column:\n {nan_by_column[nan_by_column > 0]}")

    except FileNotFoundError:
        logger.error(f"File not founded in {DATA_PATH}, check if its exist")
        raise
    except pd.errors.ParserError as e: 
        logger.error(f"Error parsing the CSV file: {e}")