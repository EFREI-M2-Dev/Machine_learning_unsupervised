# Détection d'Auberges Atypiques

Ce projet utilise un modèle non supervisé pour détecter des auberges atypiques dans un dataset d'hôtels au Japon. L'algorithme utilisé est **Isolation Forest**, qui permet d'identifier des anomalies (auberges avec des caractéristiques exceptionnelles).

## Prérequis

- `pandas`
- `scikit-learn`
- `matplotlib`

Installez-les avec la commande suivante :

```bash
pip install pandas scikit-learn matplotlib
```

## Utilisation

1. Exécutez le script Python :

```bash
python index.py
```

## Exemple de sortie

Le modèle affiche les auberges atypiques détectées, ainsi qu'une prédiction pour de nouvelles auberges si vous entrez leurs caractéristiques.
