import pandas as pd
import numpy as np
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler  # ⬅️ Tambahkan ini
from flask import Flask, request, render_template
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
# Inisialisasi Flask
app = Flask(__name__)

# Centroid awal dari hasil kmeans.py
initial_centroids = {
    2: [
        [0, 0, 0.261928, 0.262069, 0.5],
        [1, 0, 0.300126, 0.262069, 0.5]
    ],
    3: [
        [0, 0, 0.297556, 0.262069, 0],
        [0, 0, 0.210968, 0.262069, 1],
        [0, 1, 0.407144, 0.294828, 0.5]
    ],
    4: [
        [0, 0, 0.274691, 0.262069, 0],
        [0, 0, 0.208758, 0.262069, 1],
        [0, 1, 0.407144, 0.294828, 0.5],
        [1, 0, 0.363218, 0.262069, 0]
    ],
    5: [
        [0, 0, 0.241319, 0.274138, 0],
        [0, 0, 0.237981, 0.272414, 1],
        [0, 1, 0.407144, 0.294828, 0.5],
        [1, 0, 0.363218, 0.262069, 0],
        [0, 0, 0.279562, 0.267241, 0.5]
    ],
    6: [
        [0, 0, 0.241319, 0.274138, 0],
        [0, 0, 0.258862, 0.262069, 1],
        [0, 1, 0.407144, 0.294828, 0.5],
        [1, 0, 0.363218, 0.262069, 0],
        [0, 0, 0.279562, 0.267241, 0.5],
        [1, 0, 0.252323, 0.262069, 1]
    ],
    7: [
        [0, 0, 0.20754, 0.262069, 0],
        [0, 0, 0.221701, 0.262069, 1],
        [0, 1, 0.407144, 0.294828, 0.5],
        [1, 0, 0.363218, 0.262069, 0],
        [0, 0, 0.176107, 0.262069, 0.5],
        [1, 0, 0.252323, 0.262069, 1],
        [0, 0, 0.751781, 0.322414, 0.5]
    ],
    8: [
        [0, 0, 0.20754, 0.262069, 0],
        [0, 0, 0.221701, 0.262069, 1],
        [0, 1, 0.545459, 0.262069, 0],
        [1, 0, 0.363218, 0.262069, 0],
        [1, 0, 0.300126, 0.262069, 0.5],
        [0, 0, 0.751781, 0.322414, 0.5],
        [0, 1, 0.31776, 0.289655, 1],
        [0, 0, 0.176107, 0.262069, 0.5]
    ],
    9: [
        [0, 0, 0.20754, 0.262069, 0],
        [0, 0, 0.221701, 0.262069, 1],
        [0, 1, 0.545459, 0.262069, 0],
        [1, 0, 0.363218, 0.262069, 0],
        [1, 0, 0.289754, 0.262069, 0.5],
        [0, 0, 0.751781, 0.322414, 0.5],
        [0, 1, 0.27875, 0.262069, 1],
        [0, 0, 0.176107, 0.262069, 0.5],
        [1, 1, 0.565031, 0.262069, 0.5]
    ],
    10: [
        [0, 0, 0.182286, 0.262069, 0],
        [0, 0, 0.221701, 0.262069, 1],
        [0, 1, 0.545459, 0.262069, 0],
        [1, 0, 0.363218, 0.262069, 0],
        [1, 0, 0.289754, 0.262069, 0.5],
        [0, 0, 0.73983, 0.262069, 0.5],
        [0, 1, 0.27875, 0.262069, 1],
        [0, 0, 0.176107, 0.262069, 0.5],
        [1, 1, 0.565031, 0.262069, 0.5],
        [0, 0, 0.676288, 0.262069, 0]
    ],
    11: [
        [0, 0, 0.182286, 0.262069, 0],
        [0, 0, 0.221701, 0.262069, 1],
        [0, 1, 0.545459, 0.262069, 0],
        [1, 0, 0.172274, 0.262069, 0],
        [1, 0, 0.282042, 0.262069, 0.5],
        [0, 0, 0.73983, 0.262069, 0.5],
        [0, 1, 0.27875, 0.262069, 1],
        [0, 0, 0.176107, 0.262069, 0.5],
        [1, 1, 0.565031, 0.262069, 0.5],
        [0, 0, 0.676288, 0.262069, 0],
        [1, 0, 0.736764, 0.275862, 0]
    ],
    12: [
        [0, 0, 0.182286, 0.262069, 0],
        [0, 0, 0.221701, 0.262069, 1],
        [0, 1, 0.360693, 0.262069, 0.5],
        [1, 0, 0.172274, 0.262069, 0],
        [1, 0, 0.282042, 0.262069, 0.5],
        [0, 0, 0.73983, 0.262069, 0.5],
        [0, 1, 0.427979, 0.262069, 1],
        [0, 0, 0.176107, 0.262069, 0.5],
        [1, 1, 0.565031, 0.262069, 0.5],
        [0, 0, 0.676288, 0.262069, 0],
        [1, 0, 0.736764, 0.275862, 0],
        [0, 1, 0.424371, 0.262069, 0]
    ],
    13: [
        [0, 0, 0.182286, 0.262069, 0],
        [0, 0, 0.221701, 0.262069, 1],
        [0, 1, 0.360693, 0.262069, 0.5],
        [1, 0, 0.172274, 0.262069, 0],
        [1, 0, 0.282042, 0.262069, 0.5],
        [0, 0, 0.73983, 0.262069, 0.5],
        [0, 1, 0.427979, 0.262069, 1],
        [0, 0, 0.173221, 0.262069, 0.5],
        [1, 1, 0.153468, 0.262069, 0.5],
        [0, 0, 0.676288, 0.262069, 0],
        [1, 0, 0.736764, 0.275862, 0],
        [0, 1, 0.424371, 0.262069, 0],
        [1, 1, 0.748354, 0.262069, 0.5]
    ],
    14: [
        [0, 0, 0.182286, 0.262069, 0],
        [0, 0, 0.181113, 0.262069, 1],
        [0, 1, 0.360693, 0.262069, 0.5],
        [1, 0, 0.172274, 0.262069, 0],
        [1, 0, 0.282042, 0.262069, 0.5],
        [0, 0, 0.696762, 0.260345, 0.5],
        [0, 1, 0.427979, 0.262069, 1],
        [0, 0, 0.173221, 0.262069, 0.5],
        [1, 1, 0.153468, 0.262069, 0.5],
        [0, 0, 0.676288, 0.262069, 0],
        [1, 0, 0.736764, 0.275862, 0],
        [0, 1, 0.424371, 0.262069, 0],
        [1, 1, 0.748354, 0.262069, 0.5],
        [0, 0, 0.649139, 0.262069, 1]
    ],
    15: [
        [0, 0, 0.182286, 0.262069, 0],
        [0, 0, 0.156129, 0.262069, 1],
        [0, 1, 0.805854, 0.286207, 0.5],
        [1, 0, 0.172274, 0.262069, 0],
        [1, 0, 0.282042, 0.262069, 0.5],
        [0, 0, 0.679264, 0.272414, 0.5],
        [0, 1, 0.269189, 0.262069, 1],
        [0, 0, 0.172229, 0.262069, 0.5],
        [1, 1, 0.153468, 0.262069, 0.5],
        [0, 0, 0.676288, 0.262069, 0],
        [1, 0, 0.736764, 0.275862, 0],
        [0, 1, 0.348561, 0.262069, 0],
        [1, 1, 0.748354, 0.262069, 0.5],
        [0, 0, 0.478849, 0.262069, 1],
        [1, 0, 0.252323, 0.262069, 1]
    ],
    16: [
        [0, 0, 0.182286, 0.262069, 0],
        [0, 0, 0.181113, 0.262069, 1],
        [0, 1, 0.710923, 0.262069, 0.5],
        [1, 0, 0.172274, 0.262069, 0],
        [1, 0, 0.282042, 0.262069, 0.5],
        [0, 0, 0.696762, 0.260345, 0.5],
        [0, 1, 0.31776, 0.289655, 1],
        [0, 0, 0.173221, 0.262069, 0.5],
        [1, 1, 0.153468, 0.262069, 0.5],
        [0, 0, 0.676288, 0.262069, 0],
        [1, 0, 0.736764, 0.275862, 0],
        [0, 1, 0.103409, 0.262069, 0.5],
        [1, 1, 0.748354, 0.262069, 0.5],
        [0, 0, 0.649139, 0.262069, 1],
        [1, 0, 0.252323, 0.262069, 1],
        [0, 1, 0.424371, 0.262069, 0]
    ],
    17: [
        [0, 0, 0.182286, 0.262069, 0],
        [0, 0, 0.181113, 0.262069, 1],
        [0, 1, 0.710923, 0.262069, 0.5],
        [1, 0, 0.172274, 0.262069, 0],
        [1, 0, 0.198385, 0.306897, 0.5],
        [0, 0, 0.696762, 0.260345, 0.5],
        [0, 1, 0.31776, 0.289655, 1],
        [0, 0, 0.173221, 0.262069, 0.5],
        [1, 1, 0.153468, 0.262069, 0.5],
        [0, 0, 0.676288, 0.262069, 0],
        [1, 0, 0.703933, 0.262069, 0],
        [0, 1, 0.103409, 0.262069, 0.5],
        [1, 1, 0.748354, 0.262069, 0.5],
        [0, 0, 0.649139, 0.262069, 1],
        [1, 0, 0.110219, 0.434483, 1],
        [0, 1, 0.424371, 0.262069, 0],
        [1, 0, 0.665554, 0.255172, 0.5]
    ],
    18: [
        [0, 0, 0.182286, 0.262069, 0],
        [0, 0, 0.181113, 0.262069, 1],
        [0, 1, 0.805854, 0.286207, 0.5],
        [1, 0, 0.172274, 0.262069, 0],
        [1, 0, 0.198385, 0.306897, 0.5],
        [0, 0, 0.696762, 0.260345, 0.5],
        [0, 1, 0.31776, 0.289655, 1],
        [0, 0, 0.173221, 0.262069, 0.5],
        [1, 1, 0.153468, 0.262069, 0.5],
        [0, 0, 0.676288, 0.262069, 0],
        [1, 0, 0.703933, 0.262069, 0],
        [0, 1, 0.660052, 0.25, 0.5],
        [1, 1, 0.748354, 0.262069, 0.5],
        [0, 0, 0.649139, 0.262069, 1],
        [1, 0, 0.110219, 0.434483, 1],
        [0, 1, 0.424371, 0.262069, 0],
        [1, 0, 0.665554, 0.255172, 0.5],
        [0, 1, 0.103409, 0.262069, 0.5]
    ],
    19: [
        [0, 0, 0.140615, 0.262069, 0],
        [0, 0, 0.181113, 0.262069, 1],
        [0, 1, 0.805854, 0.286207, 0.5],
        [1, 0, 0.172274, 0.262069, 0],
        [1, 0, 0.198385, 0.306897, 0.5],
        [0, 0, 0.696762, 0.260345, 0.5],
        [0, 1, 0.31776, 0.289655, 1],
        [0, 0, 0.173221, 0.262069, 0.5],
        [1, 1, 0.153468, 0.262069, 0.5],
        [0, 0, 0.766528, 0.337931, 0],
        [1, 0, 0.703933, 0.262069, 0],
        [0, 1, 0.660052, 0.25, 0.5],
        [1, 1, 0.748354, 0.262069, 0.5],
        [0, 0, 0.649139, 0.262069, 1],
        [1, 0, 0.110219, 0.434483, 1],
        [0, 1, 0.424371, 0.262069, 0],
        [1, 0, 0.665554, 0.255172, 0.5],
        [0, 1, 0.103409, 0.262069, 0.5],
        [0, 0, 0.319158, 0.262069, 0]
    ],
    20: [
        [0, 0, 0.140615, 0.262069, 0],
        [0, 0, 0.122486, 0.262069, 1],
        [0, 1, 0.805854, 0.286207, 0.5],
        [1, 0, 0.172274, 0.262069, 0],
        [1, 0, 0.198385, 0.306897, 0.5],
        [0, 0, 0.696762, 0.260345, 0.5],
        [0, 1, 0.31776, 0.289655, 1],
        [0, 0, 0.173221, 0.262069, 0.5],
        [1, 1, 0.153468, 0.262069, 0.5],
        [0, 0, 0.766528, 0.337931, 0],
        [1, 0, 0.703933, 0.262069, 0],
        [0, 1, 0.660052, 0.25, 0.5],
        [1, 1, 0.748354, 0.262069, 0.5],
        [0, 0, 0.707946, 0.289655, 1],
        [1, 0, 0.110219, 0.434483, 1],
        [0, 1, 0.424371, 0.262069, 0],
        [1, 0, 0.665554, 0.255172, 0.5],
        [0, 1, 0.103409, 0.262069, 0.5],
        [0, 0, 0.319158, 0.262069, 0],
        [0, 0, 0.286958, 0.262069, 1]
    ]
}

# Load dan siapkan data dari DATA 1.xlsx
def load_and_prepare_data(file_path):
    df = pd.read_excel(file_path)
    if 'smoking_status' not in df.columns:
        df.columns = ['id', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'smoking_status']
    return df

def clean_data(df):
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())

    if 'Unknown' in df['smoking_status'].values:
        mode = df['smoking_status'].mode()[0]
        df['smoking_status'] = df['smoking_status'].replace('Unknown', mode)

    df['smoking_status'] = df['smoking_status'].map({
        'never smoked': 0.0,
        'formerly smoked': 0.5,
        'smokes': 1.0
    })
    return df

def normalize_data(df, columns):
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler

# Langkah-langkah pemrosesan
data_train = load_and_prepare_data("data/DATA 1.xlsx")
data_train = clean_data(data_train)

features = ['hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'smoking_status']
data_train_normalized, scaler = normalize_data(data_train.copy(), ['avg_glucose_level', 'bmi'])

X = data_train_normalized[features].copy()

# Simpan min-max asli untuk validasi input user
min_glucose = data_train['avg_glucose_level'].min()
max_glucose = data_train['avg_glucose_level'].max()
min_bmi = data_train['bmi'].min()
max_bmi = data_train['bmi'].max()

# Normalisasi berdasarkan rentang aktual
X = data_train[features].copy()
X['avg_glucose_level'] = (X['avg_glucose_level'] - min_glucose) / (max_glucose - min_glucose)
X['bmi'] = (X['bmi'] - min_bmi) / (max_bmi - min_bmi)

# Pastikan smoking_status sudah numerik
smoking_map = {'never smoked': 0, 'formerly smoked': 0.5, 'smokes': 1.0}
if X['smoking_status'].dtype == object:
    X['smoking_status'] = X['smoking_status'].map(smoking_map)

# Fungsi untuk menjalankan K-means
def run_kmeans_with_initial_centroids(X, initial_centroids, max_iter=100, tol=1e-4):
    results = []
    min_dbi = float('inf')
    best_k = None
    best_centroids = None
    best_labels = None
    
    X_np = X.values
    n_samples = X_np.shape[0]
    
    for k in range(2, 21):
        centroids = np.array(initial_centroids[k])
        
        for iter_num in range(max_iter):
            distances = np.linalg.norm(X_np[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
            labels = np.argmin(distances, axis=1)
            
            new_centroids = []
            new_centroid_indices = []
            empty_clusters = 0
            
            for i in range(k):
                cluster_points = X_np[labels == i]
                if len(cluster_points) > 0:
                    mean_point = cluster_points.mean(axis=0)
                    distances_to_mean = np.linalg.norm(cluster_points - mean_point, axis=1)
                    closest_idx_in_cluster = np.argmin(distances_to_mean)
                    cluster_indices = np.where(labels == i)[0]
                    closest_data_idx = cluster_indices[closest_idx_in_cluster]
                    new_centroids.append(cluster_points[closest_idx_in_cluster])
                    new_centroid_indices.append(closest_data_idx)
                else:
                    empty_clusters += 1
                    if len(new_centroids) > 0:
                        distances_to_new_centroids = np.min(
                            np.linalg.norm(X_np[:, np.newaxis, :] - np.array(new_centroids)[np.newaxis, :, :], axis=2),
                            axis=1
                        )
                    else:
                        distances_to_new_centroids = np.full(n_samples, -np.inf)
                    mask_non_centroid = np.ones(n_samples, dtype=bool)
                    mask_non_centroid[new_centroid_indices] = False
                    valid_distances = np.where(mask_non_centroid, distances_to_new_centroids, -np.inf)
                    farthest_point_idx = np.argmax(valid_distances)
                    new_centroids.append(X_np[farthest_point_idx])
                    new_centroid_indices.append(farthest_point_idx)
            
            new_centroids = np.array(new_centroids)
            
            if np.allclose(centroids, new_centroids, atol=tol):
                break
            centroids = new_centroids
        
        wcss = sum(np.sum((X_np[labels == i] - centroids[i]) ** 2) for i in range(k))
        silhouette = silhouette_score(X_np, labels) if k > 1 else 0
        try:
            dbi = davies_bouldin_score(X_np, labels) if k > 1 else float('inf')
        except:
            dbi = float('inf')
        empty_clusters_count = empty_clusters
        
        results.append({
            'k': k,
            'wcss': wcss,
            'silhouette': silhouette,
            'dbi': dbi,
            'empty_clusters': empty_clusters_count,
            'labels': labels,
            'centroids': centroids
        })
        
        if dbi < min_dbi:
            min_dbi = dbi
            best_k = k
            best_centroids = centroids
            best_labels = labels
    
    return results, best_k, best_centroids, best_labels

# Fungsi untuk menghitung jarak Euclidean
def calculate_euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Fungsi untuk memetakan klaster ke k=2 (Risiko Tinggi dan Risiko Rendah)
def map_to_k2(centroids, k, results):
    k2_centroids = np.zeros((2, centroids.shape[1]))
    cluster_mapping = np.zeros(k, dtype=int)  # 0 untuk Rendah, 1 untuk Tinggi
    
    risk_scores = []
    for centroid in centroids:
        score = (centroid[0] + centroid[1] + centroid[2] * 2 + centroid[3] * 2 + centroid[4])
        risk_scores.append(score)
    
    risk_scores = np.array(risk_scores)
    threshold = np.median(risk_scores)
    
    for i in range(k):
        if risk_scores[i] >= threshold:
            cluster_mapping[i] = 1  # Risiko Tinggi
        else:
            cluster_mapping[i] = 0  # Risiko Rendah
    
    high_risk_points = centroids[cluster_mapping == 1]
    low_risk_points = centroids[cluster_mapping == 0]
    
    if len(high_risk_points) > 0:
        k2_centroids[1] = high_risk_points.mean(axis=0)
    else:
        k2_centroids[1] = centroids[np.argmax(risk_scores)]
    
    if len(low_risk_points) > 0:
        k2_centroids[0] = low_risk_points.mean(axis=0)
    else:
        k2_centroids[0] = centroids[np.argmin(risk_scores)]
    
    k2_labels = cluster_mapping[results[k-2]['labels']]
    
    return cluster_mapping, k2_centroids, k2_labels

# Fungsi untuk menemukan klaster terdekat dan risiko untuk data baru
def find_closest_cluster_and_risk(data_point, centroids, k, cluster_mapping):
    distances = [calculate_euclidean_distance(data_point, centroid) for centroid in centroids]
    cluster_idx = np.argmin(distances)
    risk = cluster_mapping[cluster_idx]
    return cluster_idx, risk

# Route untuk halaman utama
@app.route('/')
def index():
    return render_template('form.html')

# Route untuk menangani input pengguna
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data dari form
        input_data = {
            'name': request.form.get('name', ''),
            'hypertension': float(request.form.get('hypertension', 0)),
            'heart_disease': float(request.form.get('heart_disease', 0)),
            'avg_glucose_level': float(request.form.get('avg_glucose_level', 0)),
            'bmi': float(request.form.get('bmi', 0)),
            'smoking_status': float(request.form.get('smoking_status', 0))
        }
        
        # Normalisasi menggunakan rentang training
        data_point = np.array([
            input_data['hypertension'],
            input_data['heart_disease'],
            (input_data['avg_glucose_level'] - min_glucose) / (max_glucose - min_glucose),
            (input_data['bmi'] - min_bmi) / (max_bmi - min_bmi),
            input_data['smoking_status']
        ])
        
        # Validasi input
        if input_data['hypertension'] not in [0, 1]:
            return render_template('form.html', error="Hipertensi harus Ya (1) atau Tidak (0)")
        if input_data['heart_disease'] not in [0, 1]:
            return render_template('form.html', error="Penyakit Jantung harus Ya (1) atau Tidak (0)")
        if input_data['avg_glucose_level'] < min_glucose or input_data['avg_glucose_level'] > max_glucose:
            return render_template('form.html', error=f"Kadar Glukosa harus antara {min_glucose:.2f} dan {max_glucose:.2f} mg/dL")
        if input_data['bmi'] < min_bmi or input_data['bmi'] > max_bmi:
            return render_template('form.html', error=f"BMI harus antara {min_bmi:.2f} dan {max_bmi:.2f}")
        if input_data['smoking_status'] not in [0, 0.5, 1]:
            return render_template('form.html', error="Status Merokok harus Tidak pernah (0), Mantan (0.5), atau Aktif (1)")
        if not input_data['name'].strip():
            return render_template('form.html', error="Nama Lengkap harus diisi")

        # Prediksi cluster dan risiko
        cluster, risk = find_closest_cluster_and_risk(data_point, best_centroids, best_k, cluster_mapping)

        # Format detail untuk result.html
        hypertension_text = "Ya" if input_data['hypertension'] == 1 else "Tidak"
        heart_disease_text = "Ya" if input_data['heart_disease'] == 1 else "Tidak"
        glucose_value = input_data['avg_glucose_level']  # Sudah dalam skala asli
        bmi_value = input_data['bmi']  # Sudah dalam skala asli
        smoking_text = {0: "Tidak pernah", 0.5: "Mantan", 1: "Aktif"}.get(input_data['smoking_status'], "Tidak diketahui")
        risk_label = "Tinggi" if risk == 1 else "Rendah"
        risk_summary = f"Berdasarkan data yang Anda masukkan, risiko stroke Anda berada pada tingkat {risk_label.lower()}. Silakan konsultasikan dengan dokter untuk evaluasi lebih lanjut."

        # Visualisasi: Scatter Plot (avg_glucose_level vs bmi untuk k=2)
        scatter_fig = px.scatter(
            x=[c[2] * (max_glucose - min_glucose) + min_glucose for c in k2_centroids],
            y=[c[3] * (max_bmi - min_bmi) + min_bmi for c in k2_centroids],
            color=['Risiko Tinggi' if i == 1 else 'Risiko Rendah' for i in range(2)],
            labels={'x': 'Average Glucose Level (mg/dL)', 'y': 'BMI'},
            title='Scatter Plot: Glucose vs BMI (Risiko Tinggi vs Risiko Rendah)'
        )
        scatter_fig.add_scatter(
            x=[input_data['avg_glucose_level']],
            y=[input_data['bmi']],
            mode='markers',
            marker=dict(size=15, color='red', symbol='x'),
            name='Input Data'
        )

        # Daftar fitur dan label
        features = ['hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'smoking_status']
        feature_labels = ['Hipertensi', 'Penyakit Jantung', 'Kadar Glukosa (mg/dL)', 'BMI', 'Status Merokok']

        # Denormalisasi centroid k2
        low_centroid = [
            k2_centroids[0][0],
            k2_centroids[0][1],
            k2_centroids[0][2] * (max_glucose - min_glucose) + min_glucose,
            k2_centroids[0][3] * (max_bmi - min_bmi) + min_bmi,
            k2_centroids[0][4]
        ]

        high_centroid = [
            k2_centroids[1][0],
            k2_centroids[1][1],
            k2_centroids[1][2] * (max_glucose - min_glucose) + min_glucose,
            k2_centroids[1][3] * (max_bmi - min_bmi) + min_bmi,
            k2_centroids[1][4]
        ]

        input_values = [
            input_data['hypertension'],
            input_data['heart_disease'],
            input_data['avg_glucose_level'],
            input_data['bmi'],
            input_data['smoking_status']
        ]

        # Simpan semua chart sebagai list
        bar_charts = []

        for i in range(len(features)):
            fig = go.Figure()
            fig.add_trace(go.Bar(x=['Risiko Rendah'], y=[low_centroid[i]], name='Risiko Rendah', marker_color='green'))
            fig.add_trace(go.Bar(x=['Risiko Tinggi'], y=[high_centroid[i]], name='Risiko Tinggi', marker_color='orange'))
            fig.add_trace(go.Bar(x=['Input Anda'], y=[input_values[i]], name='Input Anda', marker_color='red'))
            
            fig.update_layout(
                title=f'Perbandingan {feature_labels[i]}',
                yaxis_title='Nilai',
                barmode='group'
            )
            bar_charts.append(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
        # Konversi plot ke JSON
        scatter_json = json.dumps(scatter_fig, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template(
            'result.html',
            name=input_data['name'],
            risk_summary=risk_summary,
            hypertension=hypertension_text,
            heart_disease=heart_disease_text,
            glucose=glucose_value,
            bmi=bmi_value,
            smoking=smoking_text,
            cluster=cluster,
            risk=risk_label,
            scatter_plot=scatter_json,
            bar_charts=bar_charts   
        )
    except Exception as e:
        return render_template('form.html', error=f"Error: {str(e)}")

if __name__ == '__main__':
    # Jalankan K-Means untuk k=2 hingga k=20
    results, best_k, best_centroids, best_labels = run_kmeans_with_initial_centroids(X, initial_centroids)
    
    # Simpan hasil DBI dan metrik lainnya
    dbi_df = pd.DataFrame([{
        'k': r['k'],
        'WCSS': r['wcss'],
        'Silhouette': r['silhouette'],
        'DBI': r['dbi'],
        'Empty Clusters': r['empty_clusters']
    } for r in results])
    dbi_df.to_csv('dbi_scores.csv', index=False)

    # Petakan k terbaik ke k=2
    cluster_mapping, k2_centroids, k2_labels = map_to_k2(best_centroids, best_k, results)

    # Simpan data training dengan label cluster dan risiko
    data_train['Cluster'] = best_labels
    data_train['Risk'] = [cluster_mapping[label] for label in best_labels]
    data_train.to_csv('data_training_clustered_with_risk.csv', index=False)

    # Simpan pemetaan cluster
    mapping_df = pd.DataFrame({
        'Cluster': range(best_k),
        'Risk': [cluster_mapping[i] for i in range(best_k)]
    })
    mapping_df.to_csv('cluster_mapping.csv', index=False)

    # Simpan centroid k terbaik
    centroid_dict = {f'C{i+1}': best_centroids[i].tolist() for i in range(best_k)}
    with open('centroids_best.json', 'w') as f:
        json.dump(centroid_dict, f)

    # Simpan centroid k=2
    centroid_k2_dict = {f'C2_{i+1}': k2_centroids[i].tolist() for i in range(2)}
    with open('centroids_k2.json', 'w') as f:
        json.dump(centroid_k2_dict, f)

    # Jalankan aplikasi Flask
    app.run(debug=True)