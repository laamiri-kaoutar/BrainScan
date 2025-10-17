# BrainScan AI - Détection Automatisée des Tumeurs Cérébrales

## Contexte du Projet
BrainScan AI est une startup marocaine spécialisée dans l’imagerie médicale assistée par l’IA.  
Ce projet vise à détecter automatiquement les tumeurs cérébrales à partir d’images IRM et à assister les médecins pour un diagnostic plus rapide et précis.

---

## Structure du Projet

BrainScan_AI/
│
├─ data/                   # Contient uniquement les données brutes (images IRM)
│
├─ models/                 # Modèles entraînés sauvegardés
│
├─ notebooks/              # Notebooks d’exploration et de prétraitement
│   ├─ data_exploration.ipynb   # Exploration et prétraitement du dataset
│   └─ deployment.ipynb         # Contient la fonction predict_image et tests
│
└─ app.py                  # Interface Streamlit simple pour tester les prédictions

---

## Fonctionnalités Principales

1. Chargement et Prétraitement des Données
   - Vérification et nettoyage des images (formats acceptés : jpeg, jpg, png, bmp)
   - Redimensionnement des images à 224×224
   - Conversion en tableaux NumPy pour le modèle
   - Encodage des étiquettes
   - Division en ensembles d’entraînement et de test
   - Normalisation des pixels dans la plage [0, 1]

2. Modélisation avec CNN
   - Architecture : Conv2D + MaxPooling + Dropout + Dense
   - Compilation : Optimiseur Adam, perte categorical_crossentropy
   - Entraînement et sauvegarde du meilleur modèle

3. Évaluation
   - Courbes d’apprentissage (accuracy / loss)
   - Matrice de confusion et rapport de classification
   - Visualisation de prédictions correctes et incorrectes

4. Déploiement
   - Fonction predict_image(image_path, model, image_size=(224,224), label_encoder=None)
   - Interface Streamlit simple (app.py) pour tester les prédictions en temps réel

---

## Instructions d’Exécution

1. Installer les dépendances
```
pip install -r requirements.txt
```

2. Exécuter le Notebook
- notebooks/data_exploration.ipynb pour prétraiter les données et explorer le dataset
- notebooks/deployment.ipynb pour tester la fonction predict_image

3. Lancer l’Interface Streamlit
```
streamlit run app.py
```

4. Tester une Image
- Glisser-déposer ou sélectionner une image IRM
- Le modèle prédit automatiquement la classe de l’image

---

## Notes
- Garder les dossiers data/, models/ et notebooks/ à la racine du projet
- Tout le code est commenté pour faciliter la reproductibilité et la compréhension
