import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class PredictiveMaintenanceDataset(Dataset):
    def __init__(self, csv_path, window_size=24, forecast_horizon=12, mode='pretrain', scaler=None):
        """
        It reads the .csv file, saves how many data to look in the past (window_size) and
        how many in the future (forecast_horizon), and sets the working mode (mode = 'pretrain')
        """
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.mode = mode
        
        print(f"Loading data {csv_path}...")
        self.df = pd.read_csv(csv_path)
        
        # Feature selection
        self.feature_cols = ['volt', 'rotate', 'pressure', 'vibration', 'age', 'model']
        
        # Label finding
        self.label_cols = [col for col in self.df.columns if col.startswith('fail_')]
        
        # Normalisation handling
        continuous_cols = ['volt', 'rotate', 'pressure', 'vibration']

        # Normalization
        if scaler is None:
            self.scaler = StandardScaler()
            self.df[continuous_cols] = self.scaler.fit_transform(self.df[continuous_cols])
        else:
            self.scaler = scaler
            self.df[continuous_cols] = self.scaler.transform(self.df[continuous_cols])
        
        # Temporal windows
        print("Creating time sequences...")
        self.sequences = self._build_sequences()

    def _build_sequences(self):
        """
        Building valid indices to extract time windows;
        making sure that the data of different machines are not mixed.
        """
        sequences = []
        machines = self.df['machineID'].unique()
        
        for machine in machines:
            machine_indices = self.df[self.df['machineID'] == machine].index.values
            
            margin = self.window_size if self.mode == 'pretrain' else self.window_size + self.forecast_horizon
            max_start_idx = len(machine_indices) - margin
            
            for i in range(max_start_idx + 1):
                start_idx = machine_indices[i]
                end_idx = start_idx + self.window_size
                
                if self.mode == 'pretrain':
                    sequences.append({'start': start_idx, 'end': end_idx})
                else:
                    label_start = end_idx   # to avoid overlap between input data and the label to predict
                    label_end = end_idx + self.forecast_horizon
                    sequences.append({
                        'start': start_idx, 
                        'end': end_idx,
                        'label_start': label_start,
                        'label_end': label_end
                    })
                    
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Extract the single time window from the dataframe and convert it into tensor.
        """
        seq_info = self.sequences[idx]
        
        # Extracting X
        x_data = self.df.loc[seq_info['start'] : seq_info['end'] - 1, self.feature_cols].values
        x_tensor = torch.tensor(x_data, dtype=torch.float32)
        
        if self.mode == 'pretrain':
            # In pretrain, we divide in past and future
            half_point = self.window_size // 2
            x_past = x_tensor[:half_point]      # first 12 hours
            x_future = x_tensor[half_point:]    # last 12 hours
                                                # the task is to hide the future and trying to predict it form the past, ignoring failures
            return x_past, x_future
            
        else:
            # Extracting Y using the coordinates saved in the dictionary
            y_data = self.df.loc[seq_info['label_start'] : seq_info['label_end'] - 1, self.label_cols].values
            
            if y_data.size == 0:
                y_label = np.zeros(1)
            else:
                y_label = np.max(y_data, axis=0)
                
            y_tensor = torch.tensor(y_label, dtype=torch.float32)
            
            return x_tensor, y_tensor