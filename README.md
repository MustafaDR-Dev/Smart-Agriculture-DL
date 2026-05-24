# Projet Smart Agriculture — Detection de maladies des plantes

Projet de deep learning pour detecter les maladies des plantes a partir de photos de feuilles.
On utilise le dataset PlantVillage (38 classes, ~55k images) et deux modeles pre-entraines : ResNet50 et MobileNetV2.

## Structure du projet

```
notebooks/          -> notebook d'entrainement et comparaison des modeles
src/                -> script python pour entrainer MobileNetV2
backend/            -> API FastAPI pour la prediction
app.py              -> interface Streamlit
models/             -> modeles sauvegardes (.keras)
```

## Approche

- Transfer learning + fine-tuning sur ResNet50 et MobileNetV2
- Images 224x224, preprocess_input specifique a chaque modele
- Split 70/15/15 (train / validation / test)
- Class weights pour gerer le desequilibre entre classes
- EarlyStopping + ModelCheckpoint
- Evaluation sur le jeu de test avec accuracy, F1-score, matrice de confusion

Au final MobileNetV2 est retenu : performances comparables a ResNet50 avec 7x moins de parametres, plus adapte pour du deploiement.

## Lancer le projet

### Installation

```bash
# Créer et activer le venv Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Interface Streamlit

```bash
source .venv/bin/activate
streamlit run app.py
```

### API FastAPI

```bash
source .venv/bin/activate
uvicorn backend.main:app --reload
```

L'API expose deux endpoints :
- `GET /health` — verifier que le serveur tourne
- `POST /predict` — envoyer une image, recuperer la prediction

Exemple :
```bash
curl -X POST http://localhost:8000/predict -F "file=@photo_feuille.jpg"
```

La doc auto est sur http://localhost:8000/docs

## Dataset

PlantVillage via `tensorflow_datasets`. Pas besoin de telecharger manuellement, tfds s'en charge au premier lancement.

## Resultats

Les deux modeles sont compares dans le notebook. Les metriques detaillees (classification report, confusion matrix, courbes d'entrainement) sont visibles dans les sorties du notebook.
