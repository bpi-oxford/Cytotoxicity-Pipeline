import os
import pandas as pd
import dask.dataframe as dd
import psycopg2
from sqlalchemy import create_engine
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote_plus
from multiprocessing import cpu_count
from psycopg2 import sql
from dask.distributed import Client

class CytoDataLoader():
    def __init__(self,db_params):
        """
        Initialize the DataLoader with the database connection parameters.
        """
        self.db_params = db_params

        self.create_database_user()

        self.connection_string = self._build_connection_string()
        self.engine = create_engine(self.connection_string)

    def create_database_user(self):
        print("Attempting create UTSE DB user...")
        try:
            # Connect as the admin user
            connection = psycopg2.connect(
                dbname="postgres",
                user="postgres",
                password="password",
                host=self.db_params["host"],
                port=self.db_params["port"]
            )
            connection.autocommit = True
            cursor = connection.cursor()

            # Create the database
            dbname = self.db_params["dbname"]
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(self.db_params["dbname"])))
            print(f"Database '{dbname}' created.")

            # Create the new user
            cursor.execute(
                sql.SQL("CREATE USER {} WITH PASSWORD %s")
                .format(sql.Identifier(self.db_params["user"])),
                [self.db_params["password"]]
            )
            new_user = self.db_params["user"]
            print(f"User '{new_user}' created.")

            # Grant privileges to the new user
            cursor.execute(
                sql.SQL("GRANT ALL PRIVILEGES ON DATABASE {} TO {}")
                .format(sql.Identifier(self.db_params["dbname"]), sql.Identifier(self.db_params["user"]))
            )
            print(f"Granted privileges on '{dbname}' to '{new_user}'.")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()

    def _build_connection_string(self):
        """
        Dynamically build the database connection string from the DB_PARAMS dictionary.
        """
        return (
            f"postgresql+psycopg2://{quote_plus(self.db_params['user'])}:"
            f"{quote_plus(self.db_params['password'])}@"
            f"{self.db_params['host']}:{self.db_params['port']}/"
            f"{self.db_params['dbname']}"
        )

    def create_table_if_not_exists(self,table_name,data):
        """
        Create a table in the database if it does not exist.
        """
        with self.engine.connect() as connection:
            data.head(0).to_sql(
                name=table_name,
                con=connection,
                if_exists="replace",  # Replace with a new table structure
                index=False           # Do not include DataFrame index
            )

    def load_csv_to_table(self,csv_path, table_name):
        """
        Load single CSV file into the specified table in PostgreSQL.
        """

        try:
            # Read and combine all CSV files
            data = dd.read_csv(os.path.join(csv_path,"*.part"))
            
            # Ensure the table exists
            self.create_table_if_not_exists(table_name, data)

            #  Load data into the table
            print(f"Loading data from {csv_path} to DB table {table_name}...")
            data.to_sql(
                name=table_name,
                uri=self.connection_string,
                if_exists='replace',  # Replace the old table data
                index=False          # Do not write the DataFrame index
            )
            print(f"Successfully loaded data from {csv_path} into {table_name}.")
        except Exception as e:
            print(f"Error loading data into {table_name}: {e}")

    def load_data_parallel(self,metadata_df, max_workers=cpu_count()):
        """
        Load data from metadata into separate tables in parallel.
        """
        
        # define a function for each thread to execute
        def task(row):
            table_name = f'{row["condition"]}_{row["cell_type"]}'
            csv_path = row["path"]
            self.load_csv_to_table(csv_path, table_name)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(task,metadata_df.to_dict(orient="records"))

        print("All data loaded to DB successfully")

def main():
    client = Client()
    print(client)

    METADATA = {
        "condition": ["WT","WT","Cyto","Cyto","CancerOnly","TCellOnly"],
        "cell_type": ["cancer","tcell","cancer","tcell","cancer","tcell"],
        "path": [
            "/mnt/Data/UTSE/2023_11_18_1G4PrimCD8_WT_HCT116_CTFR_100nM_CTG_500nM_ICAM5ug_framerate10sec_flow_0p15mlperh_analysis/plots/combined/roi5/data_WT_cancer",
            "/mnt/Data/UTSE/2023_11_18_1G4PrimCD8_WT_HCT116_CTFR_100nM_CTG_500nM_ICAM5ug_framerate10sec_flow_0p15mlperh_analysis/plots/combined/roi5/data_WT_tcell",
            "/mnt/Data/UTSE/2023_11_24_1G4PrimCD8_Nyeso1_HCT116_CTFR_100nM_ICAM5ug_framerate10sec_flow_0p1mlperh_analysis/plots/combined/roi5/data_Cyto_cancer",
            "/mnt/Data/UTSE/2023_11_24_1G4PrimCD8_Nyeso1_HCT116_CTFR_100nM_ICAM5ug_framerate10sec_flow_0p1mlperh_analysis/plots/combined/roi5/data_Cyto_tcell",
            "/mnt/Data/UTSE/2023_10_03_Nyeso1_HCT116_framerate_10sec_flowrate_0p15mlperh_analysis/plots/combined/roi5/data_Cancer_cancer",
            "/mnt/Data/UTSE/2023_12_07_1G4PrimCD8_CTFR100nM_ICAM5ug_framerate10sec_flow0p1mlperh_withIL2_analysis/plots/combined/roi5/data_TCell_tcell",
        ]
    }

    metadata_df = pd.DataFrame.from_dict(METADATA)

    # %%
    DB_PARAMS = {
        "dbname": "UTSE",
        "user": "utse",
        "password": "data_hell",
        "host": "localhost",
        "port": "5432"
    }

    loader = CytoDataLoader(DB_PARAMS)
    loader.load_data_parallel(metadata_df)

if __name__ == "__main__":
    main()