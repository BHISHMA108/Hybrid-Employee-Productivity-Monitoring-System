import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# ==========================
# 1️⃣ Load Model
# ==========================
model = load_model("D:/Final Year/Sem 2/Major Project/Model Project/Project/keras_model_with_5k_images.h5")

# Compile (removes warning, optional but clean)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ==========================
# 2️⃣ Load Test Data
# ==========================
test_dir = "D:/Final Year/Sem 2/Major Project/Model Project/EfficienSee Model Dataset/valid/images"

img_size = (224, 224)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='categorical',   
    shuffle=False
)

# ==========================
# 3️⃣ Predictions
# ==========================
y_pred_prob = model.predict(test_generator)

# Convert probabilities to class index
y_pred = np.argmax(y_pred_prob, axis=1)

y_true = test_generator.classes

# ==========================
# 4️⃣ Classification Report
# ==========================
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))

# ==========================
# 5️⃣ Confusion Matrix
# ==========================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=test_generator.class_indices.keys(),
            yticklabels=test_generator.class_indices.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ==========================
# 6️⃣ ROC Curve
# ==========================

# Use probability of class 1
fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob[:,1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label="AUC = %0.2f" % roc_auc)
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

print("AUC Score:", roc_auc)

# ==========================
# 7️⃣ Overall Accuracy
# ==========================
loss, accuracy = model.evaluate(test_generator)
print("\nTest Accuracy:", accuracy)