import pandas as pd
import numpy as np

from dataclasses import dataclass
from typing import List, Tuple

from sklearn.preprocessing import StandardScaler, OneHotEncoder

@dataclass
class DataLoader:
    data: pd.DataFrame
    # Se crea un objeto StandardScaler para X
    scalerX: StandardScaler = StandardScaler()
    # Se crea un objeto StandardScaler para y
    scalery: StandardScaler = StandardScaler()
    # Se crea un objeto OneHotEncoder
    encoder: OneHotEncoder = OneHotEncoder()
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # Se identifican el número de pisos y el total de pisos
        data[["Floor_Number", "Total_Floors"]] = data["Floor"].str.extract(r'(\d+)\s*out of (\d+)').astype(float)
        return data
        
    def _category_encoding(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        # Se aplica el encoder a las columnas categóricas
        encoded_data = self.encoder.fit_transform(data[columns],)
        return encoded_data.toarray()

    def _standard_scaling_data(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        # Se aplica el scaler a las columnas numéricas
        scaled_data = self.scalerX.fit_transform(data[columns])
        return scaled_data
    
    def _standard_scaling_target(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        # Se aplica el scaler a las columnas numéricas
        scaled_data = self.scalery.fit_transform(data[columns])
        return scaled_data

    def inverse_transform_target(self, col: pd.Series) -> pd.Series:
        return self.scalery.inverse_transform(col)

    def load_and_preprocess_data(self) -> pd.DataFrame:
        data_processed = self._preprocess_data(self.data)
        X = np.hstack((
            self._category_encoding(data_processed, ["Area Type", "City", "Furnishing Status", "Tenant Preferred", "Point of Contact"]),
            self._standard_scaling_data(data_processed, ["BHK", "Size", "Bathroom", "Floor_Number", "Total_Floors"])
        ))
        y = self._standard_scaling_target(data_processed, ["Rent"])
        return X, y
    
def main():
    # Se carga el archivo de datos
    ruta_data = "C:\\Users\\User\\Documents\\U Central\\DeepLearning\\DL_T1\\data\\House_Rent_Dataset.csv"
    data = pd.read_csv(ruta_data)
    data_loader = DataLoader(data)
    X, y = data_loader.load_and_preprocess_data()
    print(f"{X.shape=}")
    print(f"{y.shape=}")
    
    print(f"{X[:5]}")
    print(f"{y[:5]}")
    y_inversed = data_loader.inverse_transform_target(y)
    print(f"{y_inversed[:5]}")
    ...
    
if __name__ == '__main__':
    main()