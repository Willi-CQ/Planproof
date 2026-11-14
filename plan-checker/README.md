# Stoa

FastAPI sert l’interface et l’API. L’analyse envoie le même PNG canonique aux moteurs déterministe et ML. Quand le visiteur demande une explication, Nemotron examine un recadrage en cas de désaccord. Si les propositions sont nombreuses ou dispersées, il examine une zone avec jusqu’à quatre propositions. Le backend vérifie que les références citées appartiennent à cette zone, puis Gemini reçoit le résumé textuel pour rédiger la réponse de l’assistant Stoa. La lecture de la zone reste à vérifier sur le plan. Le calcul du total et son statut restent du ressort de PlanProof.

## Lancer le front et l’API

1. Installer les dépendances Python de `backend/requirements.txt` et celles des deux moteurs. Le [guide de configuration des moteurs](../docs/gemini-arbitration.md) indique les variables de leurs environnements Python et du modèle ML.
2. Configurer `NVIDIA_NEMOTRON_KEY`, `GEMINI_API_KEY` et `GEMINI_MODEL` (par exemple `gemini-3.8-flash`) comme variables d’environnement du serveur. La page ne demande aucune clé.
3. À la racine, lancer `python -m uvicorn app:app --app-dir backend --port 8001` et ouvrir [Stoa](http://127.0.0.1:8001/).
4. Cliquer sur **Tester le plan d’exemple**. Sur ce cas synthétique, l’explication de Sona est demandée automatiquement si le service est configuré. Pour un plan importé, cocher **Demander une explication à Sona** avant de cliquer sur **Analyser le plan** si une revue externe est souhaitée.

Le [site GitHub Pages](https://willi-cq.github.io/Planproof/) est l’aperçu statique de la branche `front`. Le parcours avec les deux moteurs et l’assistant Stoa utilise une instance FastAPI hébergée.

Pour partager une démo de l’API, générer un mot de passe dans `STOA_ACCESS_PASSWORD`, puis lancer `python -m uvicorn hosted:app --app-dir backend --host 127.0.0.1 --port 8001` derrière un point d’entrée HTTPS. L’identifiant de la démo est `stoa`. Le navigateur demande uniquement le mot de passe de la démo.

Sur téléphone, ouvrir le lien HTTPS dans Safari ou Chrome et utiliser **Sur l’écran d’accueil** ou **Installer l’application**, selon le navigateur. Stoa fonctionne en ligne et peut redemander le mot de passe d’accès.
