# MNIST-Numpy-Architect
Un réseau de neurones à 5 couches fait entièrement en NumPy avec optimiseur Momentum. Précision de 97.5% sur MNIST. Inclut une interface de dessin Tkinter

Ce projet implémente un Perceptron Multicouche (MLP) entièrement conçu avec **NumPy**, sans utiliser les fonctionnalités de Deep Learning comme PyTorch ou TensorFlow. Le but est de démontrer la compréhension des mathématiques derrière la rétropropagation et l'optimisation.

## Performances
* Précision (Accuracy) : 97.5% sur le dataset MNIST.
* Architecture : 5 couches denses (Entrée -> 128 -> 64 -> 32 -> 16 -> Sortie).
* Optimiseur : Momentum (Gradient Descent avec inertie).

## Caractéristiques Techniques
* Initialisation : Méthode de **He Initialization** pour stabiliser les gradients.
* Fonctions d'activation : ReLU pour les couches cachées et Softmax pour la couche de sortie.
* Loss : Cross-Entropy (Entropie Croisée).
* Vectorisation : Calculs matriciels optimisés avec NumPy pour un entraînement rapide.
* Interface Graphique : Une application interactive avec **Tkinter** permettant de dessiner des chiffres et de voir les prédictions du modèle en temps réel.

## Structure du Projet
* `main.py` : Contient la classe du modèle, la logique d'entraînement et de sauvegarde.
* `test_model.py` : L'interface graphique (GUI) pour tester le modèle entraîné.
* `pois_parametre.npz` : Fichier contenant les poids et biais optimisés du modèle.
* `mniset_pytorch_cnn.py` : code du model mais entrainer avec pytorch(CNN), partique pour voir la difference entre la structure pytorch et numpy


