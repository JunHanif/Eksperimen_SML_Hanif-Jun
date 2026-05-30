import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os


def preprocess_data(file_path, save_dir):
    # Membuat folder output jika belum tersedia
    os.makedirs(save_dir, exist_ok=True)

    # membaca dataset
    df = pd.read_csv(file_path)

    # mengisi kolom 'Symptoms' dengan nilai 'No Symptoms' pada kolom yang kosong
    df['Symptoms'] = df['Symptoms'].fillna('No Symptoms')

    # menghapus kolom yang tidak relevan
    df_clean = df.drop(columns=['Patient_ID', 'Sample_Date'])

    # menghapus kolom 'Hantavirus_Type', 'Disease_Onset_Date', dan 'Disease_Severity' karena hanya muncul ketika hantavirus positif (1)
    df_clean = df_clean.drop(columns=['Hantavirus_Type', 'Disease_Onset_Date', 'Disease_Severity'])

    # menghapus data duplikat
    df_clean = df_clean.drop_duplicates()

    # encoding data kategorikal
    data_categorical = df_clean.select_dtypes(include=['object'])
    le = LabelEncoder()
    df_clean[data_categorical.columns] = data_categorical.apply(le.fit_transform)

    df_clean.to_csv(os.path.join(save_dir, 'df_clean.csv'), index=False)

    # memisahkan fitur dan target
    X = df_clean.drop(columns=['Hantavirus_Positive'])
    y = df_clean['Hantavirus_Positive']

    # split dataset menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # normalisasi data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ubah menjadi dataframe
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # simpan hasil preprocessing
    X_train_scaled_df.to_csv(os.path.join(save_dir,'X_train_scaled.csv'), index=False)
    X_test_scaled_df.to_csv(os.path.join(save_dir, 'X_test_scaled.csv'), index=False)
    y_train.to_csv(os.path.join(save_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(save_dir, 'y_test.csv'), index=False)

if __name__ == '__main__':
    file_path = 'hantavirus_dataset_raw.csv'
    save_dir = 'preprocessing/hantavirus_dataset_preprocessing'
    preprocess_data(file_path, save_dir)