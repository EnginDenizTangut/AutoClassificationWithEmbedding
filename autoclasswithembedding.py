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

zip_path = "natural_images.zip"      
output_dir = "natural_images"        
num_per_class = 100         
cluster_dir = "clusters"    
report_file = "cluster_report.txt"  

for d in [output_dir, cluster_dir, "temp_extract"]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)

print("ZIP dosyası analiz ediliyor...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_contents = zip_ref.namelist()
    print(f"ZIP içeriği ({len(zip_contents)} dosya):")
    for item in zip_contents[:20]:  
        print(f"  {item}")
    if len(zip_contents) > 20:
        print(f"  ... ve {len(zip_contents) - 20} dosya daha")

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("temp_extract")

def analyze_directory(path, level=0):
    """Dizin yapısını recursively analiz et"""
    items = []
    indent = "  " * level
    
    try:
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                print(f"{indent}📁 {item}/")
                sub_items = analyze_directory(item_path, level + 1)
                items.extend(sub_items)
            else:
                ext = os.path.splitext(item)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
                    print(f"{indent}🖼️  {item}")
                    items.append(item_path)
                else:
                    print(f"{indent}📄 {item}")
    except PermissionError:
        print(f"{indent}❌ Erişim engellendi: {path}")
    
    return items

print("\nDizin yapısı analizi:")
all_image_files = analyze_directory("temp_extract")

print(f"\nToplam {len(all_image_files)} görsel dosya bulundu.")

if len(all_image_files) == 0:
    print("\nDoğrudan görsel aranıyor...")
    all_files = []
    for root, dirs, files in os.walk("temp_extract"):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
                all_files.append(file_path)
                print(f"Görsel bulundu: {file_path}")
    
    all_image_files = all_files

if len(all_image_files) == 0:
    print("❌ HATA: Hiç görsel dosya bulunamadı!")
    print("Lütfen ZIP dosyasının görsel içerdiğinden emin olun.")
    print("Desteklenen formatlar: .jpg, .jpeg, .png, .bmp, .tiff, .gif")
    exit()

print(f"\n✅ {len(all_image_files)} görsel dosya bulundu.")

selected_files = random.sample(all_image_files, min(len(all_image_files), num_per_class * 5))
random.shuffle(selected_files)

print(f"Seçilen dosya sayısı: {len(selected_files)}")

for i, file_path in enumerate(selected_files):
    try:
        ext = os.path.splitext(file_path)[1]
        dst_path = os.path.join(output_dir, f"{i+1}{ext}")
        shutil.copy(file_path, dst_path)
        print(f"Kopyalandı: {os.path.basename(file_path)} -> {os.path.basename(dst_path)}")
    except Exception as e:
        print(f"Kopyalama hatası {file_path}: {e}")

print("\nDINOv2 modeli yükleniyor...")
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base")

def get_embedding(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")  
        with torch.no_grad():
            outputs = model(**inputs)
            emb = outputs.last_hidden_state[:, 0].squeeze().numpy()
        return emb
    except Exception as e:
        print(f"Hata embedding alırken: {img_path} -> {e}")
        return None

file_paths = glob(f"{output_dir}/*.*")
print(f"\nEmbedding çıkarılıyor: {len(file_paths)} görsel")

embeddings = []
file_names = []

for i, path in enumerate(file_paths):
    print(f"İşleniyor ({i+1}/{len(file_paths)}): {os.path.basename(path)}")
    emb = get_embedding(path)
    if emb is not None:
        embeddings.append(emb)
        file_names.append(os.path.basename(path))
    
    if (i + 1) % 10 == 0:
        print(f"  ✓ {i+1}/{len(file_paths)} işlendi")

if len(embeddings) == 0:
    raise ValueError("Hiç embedding oluşturulamadı! Dosyaları ve formatlarını kontrol et.")

embeddings = np.array(embeddings)
print(f"✅ {len(embeddings)} embedding oluşturuldu. Shape: {embeddings.shape}")

print("\nHDBSCAN clustering başlatılıyor...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean')
labels = clusterer.fit_predict(embeddings)

unique_labels = set(labels)
print(f"Clustering tamamlandı. {len(unique_labels)} cluster bulundu:")
for label in sorted(unique_labels):
    count = sum(1 for l in labels if l == label)
    if label == -1:
        print(f"  Noise: {count} görsel")
    else:
        print(f"  Cluster {label}: {count} görsel")

print("\nCluster klasörleri oluşturuluyor...")
for label in unique_labels:
    folder_name = f"cluster_{label}" if label != -1 else "noise"
    label_dir = os.path.join(cluster_dir, folder_name)
    os.makedirs(label_dir, exist_ok=True)

    count = 0
    for fname, lbl in zip(file_names, labels):
        if lbl == label:
            src_path = os.path.join(output_dir, fname)
            dst_path = os.path.join(label_dir, fname)
            shutil.copy(src_path, dst_path)
            count += 1
    
    print(f"  {folder_name}: {count} görsel")

print("\nPCA görselleştirmesi oluşturuluyor...")
pca = PCA(n_components=2).fit_transform(embeddings)
plt.figure(figsize=(12, 8))
scatter = plt.scatter(pca[:,0], pca[:,1], c=labels, cmap="tab20", s=60, alpha=0.7)
plt.colorbar(scatter, label="Cluster ID")
plt.title("DINOv2 + HDBSCAN Clustering Sonuçları")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nCluster raporu '{report_file}' oluşturuluyor...")
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("DINOv2 + HDBSCAN Clustering Raporu\n")
    f.write("="*50 + "\n\n")
    
    f.write(f"Toplam görsel sayısı: {len(file_names)}\n")
    f.write(f"Cluster sayısı: {len([l for l in unique_labels if l != -1])}\n")
    f.write(f"Noise (outlier) sayısı: {sum(1 for l in labels if l == -1)}\n\n")
    
    for label in sorted(unique_labels):
        cluster_files = [fname for fname, lbl in zip(file_names, labels) if lbl == label]
        if label == -1:
            f.write(f"NOISE ({len(cluster_files)} görsel):\n")
        else:
            f.write(f"CLUSTER {label} ({len(cluster_files)} görsel):\n")
        
        for fname in sorted(cluster_files):
            f.write(f"  - {fname}\n")
        f.write("\n")

print(f"✅ Tamamlandı!")
print(f"   - Görseller: '{output_dir}/' klasöründe")
print(f"   - Cluster klasörleri: '{cluster_dir}/' klasöründe")
print(f"   - Rapor: '{report_file}' dosyasında")

if os.path.exists("temp_extract"):
    shutil.rmtree("temp_extract")
    print("   - Geçici dosyalar temizlendi")