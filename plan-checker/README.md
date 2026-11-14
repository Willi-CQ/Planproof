# Planwise — prototype d’analyse de plans

Interface en français pour déposer un plan 2D, l’analyser avec Gemini, envoyer le plan aux deux algorithmes, rapprocher leurs JSON, expliquer les écarts dans un chatbot et solliciter JEV en complément.

## Lancer l’application

1. Installer Node.js 20 ou une version plus récente.
2. Copier `.env.example` en `.env` et renseigner les variables disponibles.
3. Lancer `node --env-file=.env server.mjs` depuis ce dossier.
4. Ouvrir `http://localhost:4173`.

Sans configuration, l’application s’ouvre en mode démonstration. Le jeu de données affiché est simulé et n’analyse pas le fichier réellement déposé. En mode réel, les trois connecteurs Gemini, déterministe et ML doivent être configurés. JEV reste facultatif.

## Intégrations

Gemini utilise l’API Interactions, avec un document PDF ou une image transmis en données base64 et une réponse structurée JSON. La clé reste dans le processus serveur. Le modèle par défaut est `gemini-3.8-flash` et peut être changé avec `GEMINI_MODEL`.

Le serveur envoie aux deux algorithmes une requête POST JSON de la forme :

```json
{
  "job_id": "plan-…",
  "file": { "name": "plan.pdf", "mime_type": "application/pdf", "data": "…base64…" },
  "gemini": { "rooms": [], "objects": {}, "uncertainties": [] }
}
```

Chaque connecteur doit répondre avec un JSON. Le contrôle croisé attend ces champs normalisés :

```json
{
  "room_count": 5,
  "area_m2": 68.2,
  "counts": { "walls": 14, "doors": 5, "windows": 4 }
}
```

Champs supplémentaires (coordonnées, boîtes de détection, métadonnées, scores) conservés tels quels et consultables dans l’onglet JSON. Les surfaces sont considérées discordantes lorsque leur différence dépasse `AREA_TOLERANCE_M2`; les autres comptes doivent être égaux. Une divergence ou une incertitude remontée par Gemini donne `review_required`. Cette règle doit être adaptée et validée par votre métier avant usage opérationnel.

L’appel JEV est un POST vers `JEV_ENDPOINT` avec `{ "analysis": { … } }`. Le résultat JSON reçu est ajouté à l’analyse. Le format exact de JEV devra être confirmé avec son fournisseur.

## Limites de cette première version

- L’adaptateur suppose que les deux algorithmes acceptent un POST JSON et renvoient les champs normalisés ci-dessus. Si leurs interfaces diffèrent, seule la couche d’adaptation de `server.mjs` est à remplacer.
- L’interface ne persiste pas les fichiers ou les conversations. Le fichier transmis est gardé en mémoire pendant la requête.
- En mode réel, le fichier est transmis à Gemini puis aux deux URLs algorithmiques configurées; vérifiez ces services et leurs règles de conservation avant d’utiliser des plans sensibles.
- Le chatbot répond à partir de l’extraction structurée de Gemini. Il ne remplace pas la validation d’un professionnel.
- La version n’intègre pas encore d’authentification, de journal d’audit, de gestion d’équipes, ni de politique de conservation des plans.

## Sources de l’intégration Gemini

- [Interactions API — guide officiel Gemini](https://ai.google.dev/gemini-api/docs/interactions-overview)
- [Référence REST Interactions API](https://ai.google.dev/api/interactions-api)
- [Traitement des documents PDF](https://ai.google.dev/gemini-api/docs/document-processing)
