# report_app.py
import streamlit as st
import numpy as np

# Page setup
st.set_page_config(page_title="BrainScan AI - Project Report", layout="wide")

# --- Project Title ---
st.title(" BrainScan AI - Project Documentation")

# --- Author / Team Info ---
st.markdown("""
**Developed by:** Kaoutar Laamiri  
**Project Type:** Deep Learning - CNN for Brain Tumor Classification  
**Dataset:** MRI Brain Tumor Dataset (Glioma, Meningioma, No Tumor, Pituitary)
""")

# --- Project Overview ---
st.header(" Project Overview")
st.markdown("""
The **BrainScan AI** project aims to classify brain MRI images into four categories of tumors: 
**glioma**, **meningioma**, **pituitary**, and **no tumor**.  
This system is built using a **Convolutional Neural Network (CNN)** trained on a dataset of MRI images.  

The goal is to assist in **early diagnosis** and **decision support** for medical experts by providing 
automated and accurate predictions.

The documentation below presents:
- The complete workflow (from data preprocessing to deployment).
- The rationale behind each step.
- The obtained performance metrics and analysis.
""")

# --- Step 2: Data Loading & Preprocessing ---
st.header(" Data Loading & Preprocessing")

st.markdown("""
In this section, we prepared the dataset by **loading, cleaning, and resizing** all MRI images.

Each subfolder in the dataset represents a distinct tumor category (e.g., *glioma*, *meningioma*, *no tumor*, *pituitary*).  
We looped through each folder, loaded every image, validated its format, resized it, and stored both the image data and its corresponding label.
""")

# --- Display the code block ---
st.code("""
images = []
labels = []

for class_name in os.listdir(raw_data_path):

    class_path = os.path.join(raw_data_path, class_name)

    if not os.path.isdir(class_path):
        continue

    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)

        # Check valid image extensions
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        try:
            image = cv2.imread(image_path)
            image = cv2.resize(image, image_size)
            images.append(image)
            labels.append(class_name)
        except Exception as exception:
            print(f"Error reading {image_path}: {exception}")

images = np.array(images)
labels = np.array(labels)
""", language="python")

# --- Add your markdown explanation below the code ---
st.markdown("""
### Explanation

1. **File Validation:**  
   Only images with valid extensions (`.png`, `.jpg`, `.jpeg`, `.bmp`) are loaded to avoid errors from non-image files.

2. **Robust Loading:**  
   We used a `try-except` block to skip unreadable or corrupted images while continuing the process smoothly.

3. **Resizing:**  
   All images are resized to **224×224 pixels** to ensure consistent input dimensions for the CNN model.

4. **Label Collection:**  
   For each image, its class label is stored alongside the data for later encoding.

5. **Conversion to NumPy Arrays:**  
   The image and label lists are converted to NumPy arrays for faster computation and compatibility with TensorFlow.

**Purpose:**  
This ensures our dataset is standardized, clean, and ready for deep learning. Resizing and validating images improves model performance and prevents crashes during training.
""")

# --- Step 3: Dataset Summary ---
st.header(" Dataset Summary")

st.markdown("""
Here is a quick overview of our dataset after loading and preprocessing:
""")

# Example summary (replace with your actual numbers)
images_shape = (5000, 224, 224, 3)  # total images × height × width × channels

labels = np.array(
    ["glioma"]*2000 + ["pituitary"]*2000 + ["meningioma"]*2000 + ["no tumor"]*2000
)

st.write(f"**Total images:** {images_shape[0]}")
st.write(f"**Image dimensions:** {images_shape[1:]} (H×W×C)")
st.write(f"**Unique labels:** {np.unique(labels)}")


# --- Step 4: Class Balancing / Data Augmentation ---
st.header(" Class Balancing & Data Augmentation")

st.markdown("""
Certain classes have fewer images than others, which may lead to model bias.  
To fix this, we applied **Data Augmentation**: random transformations (rotation, zoom, shifts, horizontal flip) are generated from existing images of the minority classes until each class reaches the same size as the largest class.

This makes the dataset **balanced and more diverse**, improving the model's ability to generalize.
""")

# Example code snippet (optional to show)
st.code("""
from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

counts = Counter(labels)
max_count = max(counts.values())

datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.1,
                             height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True)

aug_images, aug_labels = [], []

for class_name, count in counts.items():
    if count < max_count:
        indices = [i for i, lbl in enumerate(labels) if lbl == class_name]
        needed = max_count - count
        for i in range(needed):
            img = np.expand_dims(images[indices[i % len(indices)]], 0)
            aug_img = next(datagen.flow(img, batch_size=1))[0].astype(np.uint8)
            aug_images.append(aug_img)
            aug_labels.append(class_name)

images = np.concatenate([images, np.array(aug_images)], axis=0)
labels = np.concatenate([labels, np.array(aug_labels)], axis=0)
print("Balanced dataset size:", images.shape, labels.shape)
""", language="python")

# Optional: show the new class counts
from collections import Counter
st.write("**Balanced dataset size: 8000, 224, 224, 3**")
st.write("**Images per class after augmentation:**", dict(Counter(labels)))


# --- Step 5: Label Encoding & One-Hot Encoding ---
st.header("🏷️ Label & One-Hot Encoding")

st.markdown("""
- **Label Encoding:**  
  Converts categorical class names into numeric IDs.  
  Example: `"glioma" → 0, "meningioma" → 1, "no tumor" → 2, "pituitary" → 3`.  
  Each class gets a unique number for model compatibility.

- **One-Hot Encoding:**  
  Converts numeric labels into binary vectors representing classes.  
  Example: `2 → [0, 0, 1, 0]`.  
  The correct class is marked with 1 and all others with 0.  
  This is essential for multi-class classification in neural networks.

- **Purpose:**  
  Ensures that models handle categorical labels properly without misinterpreting numeric IDs as continuous values.
""")

# Example code snippet (optional to show)
st.code("""
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels_shuffled)

# Convert numeric labels to one-hot vectors
labels_categorical = to_categorical(labels_encoded)

print("Classes:", label_encoder.classes_)
print("Example encoded labels:", labels_categorical[:5])
""", language="python")

# --- Step 6: Dataset Split ---
st.header("Dataset Split")

st.markdown("""
The dataset is split into two subsets using `train_test_split`:  

- **Training set (80%)** for model learning  
- **Test set (20%)** for evaluating performance on unseen data  

The `stratify` option ensures that the class proportions remain the same in both subsets.
""")

# Example code snippet
st.code("""
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    images_shuffled, labels_categorical, 
    test_size=0.2, random_state=42, stratify=labels_shuffled
)

print("Training set:", X_train.shape, y_train.shape)
print("Test set:", X_test.shape, y_test.shape)
""", language="python")

st.write("**Training set:** (6400, 224, 224, 3), (6400, 4)")
st.write("**Test set:** (1600, 224, 224, 3), (1600, 4)")


# --- Step 7: Pixel Normalization ---
st.header("Pixel Normalization")

st.markdown("""
Images consist of pixels with values ranging from 0 to 255.  
To facilitate model learning and improve neural network stability,  
these values are normalized to the range [0, 1] by dividing each pixel by 255.  

This step **accelerates convergence** and reduces the risk of numerical instability.
""")

# Example code snippet
st.code("""
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

print("Normalized pixel range:", X_train.min(), "to", X_train.max())
""", language="python")

# Display results directly
st.write("**Normalized pixel range:** 0.0 to 1.0")


# --- Step 8: CNN Architecture ---
st.header("CNN Architecture")

st.markdown("""
The CNN model is designed to extract features from MRI images and classify them into four classes (glioma, meningioma, pituitary, no tumor).  

### Model Layers

- **Conv2D**: extracts image features like shapes and textures. Using increasing filters (32 → 64 → 128) helps the network capture more complex patterns at deeper layers.  
- **MaxPooling2D**: reduces spatial dimensions while retaining important information, which lowers computation and prevents overfitting.  
- **Dropout**: prevents overfitting by randomly deactivating neurons during training. Dropout rates increase in deeper layers (0.3 → 0.5) because deeper layers have more parameters and higher risk of overfitting.  
- **Flatten**: converts the 2D feature maps into a 1D vector to feed into dense layers.  
- **Dense**: fully connected layers perform final classification based on the extracted features.

### Activation Functions

- **ReLU** in hidden layers: fast, effective, and mitigates the vanishing gradient problem.  
- **Softmax** in the output layer: converts raw scores into probabilities for multi-class classification.

### Design Rationale

- Using **multiple Conv + Pool blocks** allows the model to progressively learn complex hierarchical features.  
- Increasing filter sizes and dropout rates in deeper layers balances learning capacity and overfitting prevention.  
- A relatively small dense layer (128 neurons) ensures the network focuses on the most relevant features before the final classification.  

Overall, this architecture is **simple, efficient, and suitable for a small to medium-sized MRI dataset**, providing a good balance between performance and computational cost.
""")

# Optional: show model summary in code
st.code("""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    Dropout(0.3),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    Dropout(0.3),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation='relu'),
    Dropout(0.4),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

model.summary()
""", language="python")


# --- Step 9: Model Compilation ---
st.header("Model Compilation")

st.markdown("""
After defining the CNN architecture, the next step is to compile the model. Compilation sets **how the model will learn** from data.  

- **Optimizer (Adam):**  
  Adam is an adaptive optimizer that combines the benefits of RMSProp and momentum. It automatically adjusts the learning rate for each weight individually based on past gradients. This helps the network converge faster and reduces the risk of getting stuck in poor local minima.

- **Learning Rate (0.001):**  
  Controls the step size during weight updates. A smaller learning rate provides more stable convergence but may take longer, while a larger rate may speed up learning but risk overshooting minima. 0.001 is a common balance for CNNs on image datasets.

- **Loss Function (categorical_crossentropy):**  
  Measures the difference between predicted probabilities and the true labels. It is specifically suited for **multi-class classification**, encouraging the model to assign high probability to the correct class.

- **Metrics (accuracy):**  
  Tracks the proportion of correctly predicted labels during training and validation, providing a simple, intuitive measure of performance.

**Why this combination works:**  
Using Adam with a moderate learning rate allows the CNN to efficiently learn hierarchical image features. The categorical cross-entropy aligns with one-hot encoded labels, and monitoring accuracy ensures we can easily detect underfitting or overfitting during training.
""")

# Example code snippet
st.code("""
from tensorflow.keras.optimizers import Adam

learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
""", language="python")


# --- Step 10: Model Summary ---
st.header("Model Summary & Parameters")

st.markdown("""
After compiling the model, we can inspect its structure and parameter counts.  
Below is a screenshot of the model summary showing each layer, output shape, and number of parameters.
""")

# Display the screenshot (replace with your actual file path)
st.image("rapport_images/image1.png", caption="CNN Model Summary", use_container_width=True)

st.markdown("""
**Why it matters:**  
Understanding the model architecture helps estimate the **complexity** and **computational requirements**.  
Convolutional layers extract features efficiently, while dense layers handle classification. Dropout layers prevent overfitting.
""")


# --- Step 11: Callbacks & Hyperparameters ---

st.header("Callbacks et Hyperparamètres")

st.code("""
# Save best model based on validation accuracy
checkpoint = ModelCheckpoint(
    '../models/best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Early stopping to prevent overfitting
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)
""", language="python")

st.markdown("""
### Entraînement contrôlé et prévention du surapprentissage

Avant l’entraînement, deux **callbacks** essentiels sont définis :

- **ModelCheckpoint**  
  Sauvegarde automatiquement le **meilleur modèle** selon la précision sur les données de validation (`val_accuracy`).  
  Cela garantit de conserver la version la plus performante du réseau, même si les résultats se dégradent ensuite.

- **EarlyStopping**  
  Interrompt l’entraînement si la **valeur de la perte de validation (`val_loss`)** ne s’améliore plus après plusieurs époques (`patience=5`).  
  L’option `restore_best_weights=True` recharge les poids du meilleur modèle observé, évitant le surapprentissage.

### Hyperparamètres

Les **hyperparamètres** déterminent la manière dont le modèle apprend :
- **Learning rate (0.001)** : vitesse de mise à jour des poids à chaque itération.  
- **Batch size** : nombre d’images traitées avant une mise à jour des poids.  
- **Epochs** : nombre de cycles complets d’entraînement sur l’ensemble des données.

Un bon réglage permet de trouver un compromis entre **vitesse de convergence**, **stabilité** et **performance**.  
Des valeurs mal choisies peuvent entraîner une convergence lente ou un modèle instable.
""")


# --- Step 12: Model Training ---

st.header("Entraînement du modèle")

st.code("""
start_time = time.time()

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[checkpoint, early_stop]
)

end_time = time.time()
print(f"Training duration: {end_time - start_time:.2f} seconds")
""", language="python")

st.markdown("""
### Entraînement du modèle

Le modèle est entraîné avec la fonction `model.fit()` :  
- Les **données d’entraînement** servent à ajuster les poids du réseau.  
- Une **partie de validation (20%)** est utilisée pour surveiller la performance sur des données non vues.  
- Les **callbacks** (EarlyStopping, ModelCheckpoint) assurent un apprentissage stable et contrôlé.  
- Les **hyperparamètres** `epochs` et `batch_size` définissent la durée et la granularité de l’entraînement.  
- Les résultats (perte et précision) sont sauvegardés dans l’objet `history` pour une future visualisation.

Cette étape constitue le cœur du processus d’apprentissage du CNN :  
le modèle apprend à reconnaître les motifs caractéristiques des tumeurs cérébrales en ajustant progressivement ses paramètres internes.
""")



# --- Step 13: Model Evaluation ---

st.header("Évaluation du modèle")

st.code("""
# Evaluate the trained model on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
""", language="python")

st.markdown("""
### Évaluation sur l’ensemble de test

Après l’entraînement, le modèle est évalué sur les **données de test** pour mesurer sa performance réelle :  
- **test_loss** : erreur moyenne sur le test  
- **test_accuracy** : proportion de prédictions correctes sur les données non vues  

Ces indicateurs permettent de vérifier la **capacité de généralisation** du modèle et de détecter un éventuel **surapprentissage**.
""")

# Display evaluation results (based on your output)
st.success(" Test Accuracy: 91.25% — Test Loss: 0.2763")

# Display the training curves (replace the path with your actual image)
st.image("rapport_images/image2.png", caption="Accuracy & Loss over Epochs", use_container_width =True)

st.markdown("""
Les graphes ci-dessus montrent l’évolution de la **précision** et de la **fonction de perte** au fil des époques.  
Une courbe de validation proche de celle de l’entraînement indique une bonne **stabilité du modèle** et une **généralisation réussie**.
""")

st.markdown("""
### Analyse des courbes d’apprentissage

Les courbes d’entraînement montrent une **hausse rapide de la précision** sur les données d’entraînement, tandis que la précision de validation atteint un plateau autour de **90 %** après quelques époques.  
La **perte de validation** commence à se stabiliser, voire à légèrement augmenter, alors que la perte d’entraînement continue de diminuer.  

**Interprétation :**  
Ces tendances indiquent un **début de surapprentissage** après environ 6 à 7 époques.  
Le modèle apprend très bien les données d’entraînement mais généralise moins bien sur les données nouvelles.  

**Conclusion :**  
Grâce à l’**Early Stopping**, l’entraînement s’arrête avant que la performance de validation ne se dégrade, assurant ainsi un bon équilibre entre apprentissage et généralisation.
""")


# --- Step 14: Confusion Matrix & Classification Report ---

st.header("Évaluation détaillée par classe")

st.markdown("""
Après l’évaluation globale, une **analyse par classe** est effectuée grâce à la **matrice de confusion** et au **rapport de classification**.
""")

# Display the confusion matrix image
st.image("rapport_images/image3.png", caption="Matrice de confusion", use_container_width =True)

st.markdown("""
### Rapport de classification

Le rapport fournit la **précision, le rappel et le score F1** pour chaque classe, permettant d’identifier les catégories plus difficiles à prédire et d’analyser les erreurs du modèle.

**Interprétation :**
- Le modèle atteint une **précision globale élevée (91%)**.  
- Certaines classes (ex. classe 1) sont légèrement plus difficiles à prédire, suggérant des caractéristiques visuelles proches d’autres catégories.  
- La matrice de confusion montre que la majorité des erreurs se concentrent entre certaines classes spécifiques, ce qui aide à cibler des améliorations futures.
""")


st.markdown("""
### Analyse des résultats par classe

- Le modèle atteint une **précision globale de 91%**, ce qui confirme une bonne généralisation sur les données de test.  
- Les classes **2** (notumor) et **3** (pituitary) sont **très bien reconnues**, avec des scores F1 supérieurs à 0.94.  
- La classe **1** (meningioma) présente une performance légèrement inférieure (F1 = 0.85), ce qui indique que certaines images peuvent être **plus ambiguës** ou proches visuellement d’autres classes.  
- La matrice de confusion montre que la plupart des erreurs concernent les classes **0 et 1**, ce qui suggère une **confusion modérée** entre ces deux catégories.  

**Conclusion :**  
Le modèle est globalement performant, mais des améliorations ciblées pourraient être apportées pour réduire les erreurs entre les classes les plus confondues.
""")


# --- Step 15: Visualisation des Prédictions ---

st.header("Visualisation des Prédictions Correctes et Incorrectes")

st.markdown("""
Cette étape permet d’évaluer visuellement les performances du modèle sur des exemples réels issus de l’ensemble de test.  

- **T (True)** : classe réelle de l’image.  
- **P (Predicted)** : classe prédite par le modèle.  
- Les titres verts indiquent des **prédictions correctes** et les titres rouges des **prédictions incorrectes**.  
- L’analyse visuelle aide à comprendre si les erreurs proviennent d’un manque de données, d’une similarité visuelle entre certaines classes ou d’un surapprentissage.
""")

# Display the screenshot of the predictions
st.image("rapport_images/image4.png", caption="Exemples de prédictions (Vert = Correct, Rouge = Incorrect)", use_container_width =True)

st.markdown("""
### Interprétation des résultats

- L’image montre un mélange de **prédictions correctes et incorrectes**.  
- **Prédictions correctes (Vert)** : le modèle identifie correctement des classes comme *notumor*, *glioma*, ou *pituitary*.  
- **Prédictions incorrectes (Rouge)** : certaines confusions apparaissent entre *meningioma* et *glioma* ou *meningioma* et *pituitary*.  

**Conclusion :**  
Cette visualisation permet de détecter les points faibles du modèle, en particulier les classes sujettes à confusion, et guide les futures améliorations :  
- collecte de plus de données pour les classes confondues,  
- ajustement de l’architecture,  
- ou réajustement des hyperparamètres.
""")

# --- Step 16: Sauvegarde du modèle entraîné ---

st.header("Sauvegarde du Modèle Entraîné")

st.code("""
model.save('../models/model_cnn.keras')
""", language="python")

st.markdown("""
Une fois le modèle entraîné et évalué, il est essentiel de le **sauvegarder** pour éviter de devoir recommencer tout l’entraînement.  

- Le modèle est enregistré au format **`.keras`** (standard Keras/TensorFlow).  
- La sauvegarde contient :  
  - Les **poids** du modèle (weights)  
  - L’**architecture complète** du réseau  
  - La **configuration de l’optimiseur**

Cette sauvegarde permet de **recharger le modèle ultérieurement** pour effectuer des prédictions ou pour un déploiement via une interface comme Streamlit, sans réentraîner le modèle.
""")


# --- Step 17: Prédiction sur une image individuelle ---

st.header("Prédiction d’une Image Individuelle")

st.markdown("""
Après l’entraînement, nous pouvons utiliser le modèle pour **prédire la classe d’une nouvelle image MRI**.  

Le processus comprend :
- Chargement du modèle entraîné et des classes (`LabelEncoder`)  
- Lecture et redimensionnement de l’image à **224×224 pixels**  
- Normalisation des pixels pour correspondre aux valeurs d’entraînement  
- Prédiction via le modèle CNN et conversion de l’indice prédit en **nom de classe**
""")

st.code("""
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import cv2

# Charger le modèle et les classes
model = load_model('./models/model_cnn.keras')
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('./models/classes.npy', allow_pickle=True)

def predict_image(image_path, model, image_size=(224,224), label_encoder=None):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.resize(image, image_size)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    class_index = np.argmax(prediction, axis=1)[0]
    if label_encoder:
        class_label = label_encoder.inverse_transform([class_index])[0]
    else:
        class_label = class_index
    return class_label

image_path = './data/raw_data/glioma/Te-gl_0022.jpg'
predicted_class = predict_image(image_path, model, label_encoder=label_encoder)
""", language="python")

st.markdown("""
**Résultat attendu :**  
Le modèle renvoie le nom de la classe prédite, par exemple : **glioma**, **meningioma**, **pituitary**, ou **notumor**.

Cette fonctionnalité peut être étendue dans Streamlit pour **téléverser n’importe quelle image et obtenir sa prédiction en temps réel**, ce qui est pratique pour le déploiement clinique ou pour tester de nouvelles données.
""")

# --- Step 18: Application Interactive de Prédiction ---

st.header("Application Interactive de Prédiction")

st.markdown("""
Cette partie présente l’**interface interactive** construite avec Streamlit pour tester le modèle sur de nouvelles images MRI.

**Fonctionnalités :**
- Téléversement d’une image via le widget `file_uploader`.  
- Conversion et affichage de l’image téléversée.  
- Prédiction de la classe tumorale à l’aide du modèle CNN entraîné.  
- Affichage de la classe prédite en temps réel.

Cette application permet à un utilisateur non technique (ex. médecin ou chercheur) de **tester rapidement le modèle** sans exécuter du code Python.
""")

# Display a screenshot of the app
st.image("rapport_images/image5.png", caption="Interface Streamlit pour la prédiction en temps réel", use_container_width =True)
