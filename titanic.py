# Importation des bibliothèques
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Chargement des données
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Exploration des données
print(train_data.head())  # Affiche les premières lignes du jeu de données d'entraînement

# Prétraitement des données
# Suppression des colonnes inutiles
train_data = train_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
test_data = test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Remplir les valeurs manquantes de la colonne 'Age' avec la moyenne
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)

# Remplir les valeurs manquantes de la colonne 'Embarked' avec la valeur la plus fréquente
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

# Encodage des variables catégorielles
label_encoder = LabelEncoder()
train_data['Sex'] = label_encoder.fit_transform(train_data['Sex'])

# Vérifier si la colonne 'Sex' existe dans les données de test
if 'Sex' in test_data.columns:
    test_data['Sex'] = label_encoder.transform(test_data['Sex'])


# Convertir les variables catégorielles en variables numériques
train_data = pd.get_dummies(train_data, columns=['Embarked'])
test_data = pd.get_dummies(test_data, columns=['Embarked'])

# Division des données en ensembles d'entraînement et de validation
X = train_data.drop('Survived', axis=1)
y = train_data['Survived']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Construction du modèle
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Évaluation du modèle
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print('Accuracy: {:.2f}%'.format(accuracy * 100))

# Prédictions sur l'ensemble de test
test_data = test_data.fillna(test_data.mean())
predictions = model.predict(test_data)

# Création du fichier de soumission
submission = pd.DataFrame({'PassengerId': pd.read_csv('test.csv')['PassengerId'], 'Survived': predictions})
submission.to_csv('submission.csv', index=False)
