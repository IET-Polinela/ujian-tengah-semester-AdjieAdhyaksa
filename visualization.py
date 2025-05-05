import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Memuat dataset yang sudah diproses
data = np.load('processed_data.npz')
X = data['X']
y = data['y']
feature_names = data['feature_names']

# 1. Distribusi Label Target (Stroke)
plt.figure(figsize=(6, 4))
sns.countplot(x=y)
plt.title("Distribusi Label Target (Stroke)")
plt.xlabel("Stroke (1 = Ya, 0 = Tidak)")
plt.ylabel("Jumlah Pasien")
plt.savefig('plot_label_distribution.png')  # Simpan gambar
plt.close()

# 2. Heatmap Korelasi
df = pd.DataFrame(X, columns=feature_names)
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Heatmap Korelasi Antara Fitur")
plt.savefig('correlation_heatmap.png')  # Simpan gambar
plt.close()

# 3. Decision Tree Visualization
# Membagi data untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melatih model Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Visualisasi Pohon Keputusan
plt.figure(figsize=(12, 8))
plot_tree(dt_model, feature_names=feature_names, class_names=['No Stroke', 'Stroke'], filled=True, rounded=True)
plt.title("Visualisasi Pohon Keputusan")
plt.savefig('decision_tree.png')  # Simpan gambar
plt.close()

# 4. Confusion Matrix
y_pred = dt_model.predict(X_test)

# Membuat confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Stroke', 'Stroke'])

# Visualisasi Confusion Matrix
plt.figure(figsize=(6, 6))
cm_display.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig('confusion_matrix.png')  # Simpan gambar
plt.close()

print("Visualisasi telah disimpan dalam file gambar.")
