import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import sys
import os

# Menambahkan path dari direktori saat ini ke sys.path agar impor modular berfungsi
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Menggunakan implementasi PyTorch yang baru
from ml_models import autoencoder_pytorch

class AnomalyDetector:
    """
    Sebuah class untuk membungkus seluruh alur kerja deteksi anomali,
    mulai dari pemrosesan data, pelatihan, hingga prediksi.
    """
    def __init__(self, model_path='checkpoints'):
        """
        Inisialisasi detector.
        :param model_path: Path untuk menyimpan atau memuat model.
        """
        # Memastikan path adalah absolut
        if not os.path.isabs(model_path):
            # Membuat path absolut relatif terhadap file anomaly_detector.py ini
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.model_path = os.path.join(base_dir, model_path)
        else:
            self.model_path = model_path
            
        self.autoencoder_model = None
        self.classifier_model = None
        self.scaler = MinMaxScaler()
        # Kita akan menyimpan encoder untuk setiap fitur kategorikal
        self.label_encoders = {}

    def preprocess_data(self, raw_data_dict, is_training=False):
        """
        Mengubah data mentah dari MikroTik (dalam format dict) menjadi DataFrame
        yang siap untuk model.
        
        :param raw_data_dict: Dictionary berisi 'connections', 'firewall_rules', 'interfaces'.
        :param is_training: Boolean, apakah ini untuk pelatihan (fit scaler/encoder) atau tidak.
        :return: DataFrame yang sudah diproses.
        """
        connections = raw_data_dict.get('connections', [])
        interfaces = raw_data_dict.get('interfaces', [])

        # --- Fitur dari Koneksi ---
        total_connections = len(connections)
        
        # Hitung koneksi per protokol
        protocols = [conn.get('protocol') for conn in connections]
        tcp_connections = protocols.count('tcp')
        udp_connections = protocols.count('udp')
        icmp_connections = protocols.count('icmp')
        
        # Hitung koneksi berdasarkan state TCP (indikator serangan SYN flood)
        tcp_states = [conn.get('tcp-state') for conn in connections if conn.get('protocol') == 'tcp']
        syn_sent_connections = tcp_states.count('syn-sent')
        established_connections = tcp_states.count('established')

        # --- Fitur dari Antarmuka ---
        # Asumsikan interface WAN memiliki 'ether1' dalam namanya atau merupakan default route
        # Untuk sekarang, kita agregat semua interface untuk kesederhanaan
        total_rx_bytes = sum(int(iface.get('rx-byte', 0)) for iface in interfaces)
        total_tx_bytes = sum(int(iface.get('tx-byte', 0)) for iface in interfaces)
        total_rx_packets = sum(int(iface.get('rx-packet', 0)) for iface in interfaces)
        total_tx_packets = sum(int(iface.get('tx-packet', 0)) for iface in interfaces)
        total_rx_drops = sum(int(iface.get('rx-drop', 0)) for iface in interfaces)

        # Membuat dictionary dari fitur yang diekstrak
        feature_dict = {
            'total_connections': total_connections,
            'tcp_connections': tcp_connections,
            'udp_connections': udp_connections,
            'icmp_connections': icmp_connections,
            'syn_sent_connections': syn_sent_connections,
            'established_connections': established_connections,
            'total_rx_bytes': total_rx_bytes,
            'total_tx_bytes': total_tx_bytes,
            'total_rx_packets': total_rx_packets,
            'total_tx_packets': total_tx_packets,
            'total_rx_drops': total_rx_drops,
        }

        # Mengubah dictionary menjadi DataFrame
        df = pd.DataFrame([feature_dict])

        # Scaling fitur
        # Saat pelatihan, kita 'fit' dan 'transform'. Saat prediksi, kita hanya 'transform'.
        if is_training:
            # fit_transform mengharapkan data 2D, jadi kita panggil pada seluruh DataFrame
            scaled_features = self.scaler.fit_transform(df)
        else:
            # Pastikan scaler sudah di-fit sebelumnya
            if not hasattr(self.scaler, 'scale_'):
                # Jika scaler belum di-fit (misalnya, saat prediksi pertama kali tanpa load model),
                # kita tidak bisa melakukan transform. Ini harus ditangani di alur kerja utama.
                # Untuk sekarang, kita lewati scaling jika belum siap.
                print("Warning: Scaler has not been fitted. Skipping scaling.")
                scaled_features = df.values
            else:
                scaled_features = self.scaler.transform(df)

        # Mengembalikan sebagai DataFrame dengan nama kolom yang sama
        df_scaled = pd.DataFrame(scaled_features, columns=df.columns)
            
        print(f"Data preprocessing complete. {len(df.columns)} features generated.")
        return df_scaled

    def train(self, list_of_normal_data, list_of_anomaly_data=None, epochs=10, batch_size=32, learning_rate=1e-3):
        """
        Melatih model Autoencoder dan Classifier menggunakan PyTorch.
        """
        print("Starting model training with PyTorch...")
        
        # 1. Preprocess data normal
        print(f"Processing {len(list_of_normal_data)} normal data points...")
        df_normal = pd.concat([self.preprocess_data(d, is_training=True) for d in list_of_normal_data], ignore_index=True)
        
        # Konversi ke PyTorch Tensor
        tensor_normal = torch.tensor(df_normal.values, dtype=torch.float32)
        dataset = TensorDataset(tensor_normal, tensor_normal)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 2. Inisialisasi dan latih Autoencoder
        num_features = df_normal.shape[1]
        self.autoencoder_model = autoencoder_pytorch.AutoEncoder(num_features)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.autoencoder_model.parameters(), lr=learning_rate)

        print(f"Training PyTorch Autoencoder for {epochs} epochs...")
        for epoch in range(epochs):
            for data in dataloader:
                inputs, _ = data
                outputs = self.autoencoder_model(inputs)
                loss = criterion(outputs, inputs)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

        # 3. Dapatkan latent representation
        print("Generating latent representations...")
        self.autoencoder_model.eval() # Set model ke mode evaluasi
        with torch.no_grad():
            normal_rep_tensor = self.autoencoder_model.get_latent_representation(tensor_normal)
            normal_rep = normal_rep_tensor.numpy()

        # 4. Siapkan data untuk classifier
        if list_of_anomaly_data:
            print(f"Processing {len(list_of_anomaly_data)} anomaly data points...")
            df_anomaly = pd.concat([self.preprocess_data(d, is_training=False) for d in list_of_anomaly_data], ignore_index=True)
            tensor_anomaly = torch.tensor(df_anomaly.values, dtype=torch.float32)
            with torch.no_grad():
                anomaly_rep_tensor = self.autoencoder_model.get_latent_representation(tensor_anomaly)
                anomaly_rep = anomaly_rep_tensor.numpy()
        else:
            print("Generating synthetic anomalies for classifier training.")
            noise = np.random.normal(loc=0.5, scale=0.5, size=normal_rep.shape)
            anomaly_rep = normal_rep + noise

        X = np.append(normal_rep, anomaly_rep, axis=0)
        y = np.append(np.zeros(len(normal_rep)), np.ones(len(anomaly_rep)))

        # 5. Latih Classifier
        print("Training a Logistic Regression classifier...")
        train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.25, random_state=42)
        clf = LogisticRegression(solver="lbfgs", max_iter=1000).fit(train_x, train_y)
        self.classifier_model = clf
        
        accuracy = np.mean(clf.predict(val_x) == val_y)
        print(f"Classifier accuracy on validation set: {accuracy:.2%}")
        
        # 6. Simpan model
        self.save_model()
        
        print("Training complete and models saved.")
        return {"status": "success", "message": f"Models trained with classifier accuracy: {accuracy:.2%}"}

    def predict(self, new_data_dict):
        """
        Membuat prediksi pada data baru menggunakan model PyTorch.
        """
        if not self.autoencoder_model or not self.classifier_model:
            print("Models not loaded. Attempting to load...")
            if not self.load_model():
                 return {"status": "error", "message": "Models are not trained or loaded."}

        # 1. Preprocess data baru
        df_processed = self.preprocess_data(new_data_dict, is_training=False)
        tensor_processed = torch.tensor(df_processed.values, dtype=torch.float32)
        
        # 2. Dapatkan latent representation
        self.autoencoder_model.eval()
        with torch.no_grad():
            latent_rep_tensor = self.autoencoder_model.get_latent_representation(tensor_processed)
            latent_rep = latent_rep_tensor.numpy()
        
        # 3. Dapatkan prediksi dari classifier
        prediction = self.classifier_model.predict(latent_rep)
        probability = self.classifier_model.predict_proba(latent_rep)
        
        is_anomaly = bool(prediction[0] == 1)
        # Mengubah prediction[0] menjadi integer agar bisa digunakan sebagai indeks
        confidence_index = int(prediction[0])
        confidence = float(probability[0][confidence_index])
        
        return {
            "is_anomaly": is_anomaly,
            "confidence": f"{confidence:.2%}",
            "model_version": "1.0.0_pytorch"
        }

    def save_model(self):
        """Menyimpan model PyTorch, classifier, dan scaler."""
        import os
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            
        print(f"Saving PyTorch models to {self.model_path}...")
        
        if self.autoencoder_model:
            torch.save(self.autoencoder_model.state_dict(), os.path.join(self.model_path, 'autoencoder.pth'))

        if self.classifier_model:
            with open(os.path.join(self.model_path, 'classifier.pkl'), 'wb') as f:
                pickle.dump(self.classifier_model, f)

        with open(os.path.join(self.model_path, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
            
        print("Models saved successfully.")

    def load_model(self):
        """Memuat model PyTorch, classifier, dan scaler."""
        import os
        
        print(f"Loading PyTorch models from {self.model_path}...")
        
        try:
            # Muat Scaler
            scaler_path = os.path.join(self.model_path, 'scaler.pkl')
            if not os.path.exists(scaler_path):
                print("Warning: Scaler file not found.")
                return False
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

            # Muat Classifier
            classifier_path = os.path.join(self.model_path, 'classifier.pkl')
            if not os.path.exists(classifier_path):
                print("Warning: Classifier model file not found.")
                return False
            with open(classifier_path, 'rb') as f:
                self.classifier_model = pickle.load(f)

            # Inisialisasi dan muat Autoencoder
            # Kita perlu tahu input_dim, yang bisa kita dapatkan dari scaler
            input_dim = self.scaler.n_features_in_
            self.autoencoder_model = autoencoder_pytorch.AutoEncoder(input_dim)
            autoencoder_path = os.path.join(self.model_path, 'autoencoder.pth')
            if not os.path.exists(autoencoder_path):
                print("Warning: Autoencoder model file not found.")
                return False
            self.autoencoder_model.load_state_dict(torch.load(autoencoder_path))
            self.autoencoder_model.eval() # Set ke mode evaluasi
                            
            print("Models loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            self.autoencoder_model = None
            self.classifier_model = None
            return False
