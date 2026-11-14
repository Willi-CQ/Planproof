# Stoa

FastAPI sert l’interface et l’API. L’analyse envoie le même PNG canonique aux moteurs déterministe et ML. En cas de désaccord, le visiteur peut autoriser NVIDIA Nemotron à examiner le recadrage. Le total reste à vérifier après un avis.

## Lancer le front et l’API

1. Installer les dépendances Python de `backend/requirements.txt` et celles des deux moteurs. Le [guide de configuration des moteurs](../docs/gemini-arbitration.md) indique les variables de leurs environnements Python et du modèle ML.
2. Configurer `NVIDIA_NEMOTRON_KEY` dans l’environnement du serveur. La clé est lue au démarrage et la page ne demande aucune clé.
3. À la racine, lancer `python -m uvicorn app:app --app-dir backend --port 8001` et ouvrir [Stoa](http://127.0.0.1:8001/).
4. Autoriser l’examen de la zone, puis cliquer sur **Tester le plan d’exemple**. Pour utiliser les deux moteurs sur un plan, importer un PDF, PNG ou JPEG et cliquer sur **Analyser le plan**.

Le [site GitHub Pages](https://willi-cq.github.io/Planproof/) est l’aperçu statique de la branche `front`. Le parcours avec les moteurs et Nemotron utilise une instance FastAPI hébergée.

Pour partager une démo de l’API, générer un mot de passe dans `STOA_ACCESS_PASSWORD`, puis lancer `python -m uvicorn hosted:app --app-dir backend --host 127.0.0.1 --port 8001` derrière un point d’entrée HTTPS. L’identifiant de la démo est `stoa`. Le navigateur demande uniquement le mot de passe de la démo.
