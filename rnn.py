import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler

# 1. LOAD DATA
print("Loading data...")
df = pd.read_csv("ADNI_Integrated_Master.csv")
print(f"Initial Rows: {len(df)}")

# 2. FILL GAPS (IMPUTATION)
# Critical Step: We sort by Patient and Time, then 'forward fill' missing values.
# If they have an amyloid scan at t=0, we carry that value forward to t=1, t=2, etc.
df = df.sort_values(['RID', 'Years'])
features = ['CENTILOIDS', 'GFAP_Q', 'aHV']

# Group by RID and ffill (propagate last valid observation forward)
df[features] = df.groupby('RID')[features].ffill()
# bfill (backward fill) handles cases where the first visit might be missing a value
df[features] = df.groupby('RID')[features].bfill()

# Now drop rows that are STILL missing data (e.g., patients who NEVER had a GFAP test)
df = df.dropna(subset=['Years'] + features)
print(f"Rows after filling gaps & dropping empties: {len(df)}")

# Filter for patients with enough history
# We need at least 3 visits (Seq Length 2 + Target 1)
counts = df['RID'].value_counts()
valid_rids = counts[counts >= 3].index
df = df[df['RID'].isin(valid_rids)]
print(f"Patients with 3+ visits: {len(valid_rids)}")

if len(df) == 0:
    print("\nCRITICAL ERROR: No data left. Check your CSV column names or merge process.")
    print("Columns in CSV:", df.columns.tolist())
    exit()

# 3. NORMALIZE
scaler = MinMaxScaler()
df[['sc_Amyloid', 'sc_Inflam', 'sc_Neuronal']] = scaler.fit_transform(df[features])

# 4. PREPARE SEQUENCES
def create_sequences(data, rid_col, seq_length=2):
    X = []
    y = []
    
    for rid, group in data.groupby(rid_col):
        # Ensure strict chronological order
        group = group.sort_values('Years')
        
        # We need continuous data. Since we ffilled, we assume continuity.
        vals = group[['sc_Amyloid', 'sc_Inflam', 'sc_Neuronal']].values
        
        if len(vals) > seq_length:
            for i in range(len(vals) - seq_length):
                X.append(vals[i:i+seq_length])
                y.append(vals[i+seq_length])
                
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(df, 'RID', seq_length=2)

print(f"Training Shapes: Inputs {X_train.shape}, Targets {y_train.shape}")

# Double check we actually have data before training
if len(X_train) == 0:
    print("Error: Sequences are empty. Try reducing 'seq_length' to 1 or checking imputation.")
    exit()

# 5. BUILD RNN
model = Sequential()
# Use Input(shape) as the first layer to avoid the UserWarning
model.add(Input(shape=(2, 3))) 
model.add(LSTM(64, return_sequences=True)) 
model.add(Dropout(0.2))
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(3, activation='linear')) 

model.compile(optimizer='adam', loss='mse')

# 6. TRAIN
# Reduced validation_split to 0.1 to save more data for training
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.1)

# 7. PREDICT (Sanity Check)
predictions = model.predict(X_train)
real_preds = scaler.inverse_transform(predictions)
real_targets = scaler.inverse_transform(y_train)

print("\n--- SAMPLE PREDICTION (Next Visit) ---")
print(f"Predicted Neuronal Volume (Ratio): {real_preds[0][2]:.5f}")
print(f"Actual Neuronal Volume (Ratio):    {real_targets[0][2]:.5f}")