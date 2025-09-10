import os
import zipfile
import random
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import hdbscan
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from glob import glob

zip_path = "images.zip"      
output_dir = "images"        
num_per_class = 200          
cluster_dir = "clusters"     
report_file = "cluster_report.txt"  

for d in [output_dir, cluster_dir, "temp_extract"]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("temp_extract")

subfolders = [f for f in os.listdir("temp_extract") if os.path.isdir(os.path.join("temp_extract", f))]

all_selected_files = []

for folder in subfolders:
    folder_path = os.path.join("temp_extract", folder)
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
             if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.jpg','.jpeg','.png'))]
    if len(files) == 0:
        print(f"Uyarı: {folder} klasöründe hiç uygun görsel yok.")
        continue
    selected = random.sample(files, min(num_per_class, len(files)))
    all_selected_files.extend(selected)

if len(all_selected_files) == 0:
    raise ValueError("Hiç görsel seçilmedi! ZIP içeriğini ve uzantıları kontrol et.")

random.shuffle(all_selected_files)

for i, file_path in enumerate(all_selected_files):
    ext = os.path.splitext(file_path)[1]
    shutil.copy(file_path, os.path.join(output_dir, f"{i+1}{ext}"))

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base")

def get_embedding(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return emb
    except Exception as e:
        print(f"Hata embedding alırken: {img_path} -> {e}")
        return None

file_paths = glob(f"{output_dir}/*.*")
print(f"{len(file_paths)} görsel seçildi.")

embeddings = []
file_names = []

for path in file_paths:
    emb = get_embedding(path)
    if emb is not None:
        embeddings.append(emb)
        file_names.append(os.path.basename(path))

if len(embeddings) == 0:
    raise ValueError("Hiç embedding oluşturulamadı! Dosyaları ve formatlarını kontrol et.")

embeddings = np.array(embeddings)

clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean')
labels = clusterer.fit_predict(embeddings)

for label in set(labels):
    folder_name = f"cluster_{label}" if label != -1 else "noise"
    label_dir = os.path.join(cluster_dir, folder_name)
    os.makedirs(label_dir, exist_ok=True)

    for fname, lbl in zip(file_names, labels):
        if lbl == label:
            src_path = os.path.join(output_dir, fname)
            dst_path = os.path.join(label_dir, fname)
            shutil.copy(src_path, dst_path)

pca = PCA(n_components=2).fit_transform(embeddings)
plt.figure(figsize=(8,6))
plt.scatter(pca[:,0], pca[:,1], c=labels, cmap="tab20", s=60)
plt.colorbar(label="Cluster ID")
plt.title("DINOv2 + HDBSCAN Clustering")
plt.show()

with open(report_file, 'w') as f:
    for fname, lbl in zip(file_names, labels):
        f.write(f"{fname} -> Cluster {lbl}\n")

print(f"Tüm görseller '{output_dir}' içinde, cluster klasörleri '{cluster_dir}' içinde kaydedildi.")
print(f"Cluster raporu '{report_file}' olarak oluşturuldu.")

shutil.rmtree("temp_extract")