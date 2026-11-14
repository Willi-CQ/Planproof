# Contrat API pour la visionneuse

Le frontend n'appelle que le backend FastAPI. Deux requêtes suffisent pour analyser un plan. Le backend lance lui-même le moteur déterministe. Le ML reste facultatif tant qu'il ne tourne pas sur la machine de démo.

## Lancer le backend avec le moteur déterministe

Depuis la racine du dépôt, avec Python 3.11 ou plus récent :

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r backend/requirements.txt -e deterministic
.venv\Scripts\python -m uvicorn app:app --app-dir backend --port 8000
```

Sans l'installation de `deterministic`, l'API répond quand même et signale `engines.deterministic: "not_installed"`. Les origines `http://localhost:<port>` et `http://127.0.0.1:<port>` sont autorisées par CORS.

## 1. Envoyer le plan

`POST /api/documents`, multipart : `file` (PDF, PNG ou JPEG, 10 Mio maximum) et `page` facultatif à partir de 1.

```json
{"document_id": "5f0c…", "page": 1, "image_url": "/api/documents/5f0c…/image", "width": 1600, "height": 800}
```

Afficher `image_url` tel quel. Toutes les boîtes renvoyées ensuite sont en pixels de cette image, origine en haut à gauche, ordre `[x0, y0, x1, y1]`. Si l'image est redimensionnée à l'écran, multiplier les boîtes par `largeur affichée / width`.

## 2. Lancer l'analyse des portes

`POST /api/v1/documents/{document_id}/door-analysis`, JSON :

```json
{"ml_detections": null, "allow_gemini_review": false}
```

- `ml_detections` : le `detections.json` brut produit par `planproof-ml infer` sur le PNG de `image_url`, lancé sans `--document-id`. Omettre le champ si le ML n'a pas tourné.
- `allow_gemini_review` : `true` seulement si l'utilisateur accepte l'envoi d'un recadrage du plan à Gemini.

Réponse (champs utiles à l'affichage) :

| Champ | Affichage |
| --- | --- |
| `status` | `SUPPORTED` : nombre confirmé. `NEEDS_REVIEW` : désaccord localisé. `INSUFFICIENT_EVIDENCE` : pas de nombre. |
| `answer` | Nombre de portes, ou `null`. N'afficher un nombre que s'il n'est pas `null`. |
| `reasons` | `analyzer_unavailable`, `non_independent`, `ambiguous_matches`, `unmatched_detections`, `coverage_unknown`. |
| `matched_evidence` | Boîtes confirmées par les deux moteurs. |
| `disputed_evidence` | Boîtes en désaccord, à mettre en évidence. |
| `unverified_evidence` | Propositions d'un seul moteur, à présenter comme non vérifiées. |
| `deterministic`, `ml` | Résumé par moteur : `status`, `version`, `detected_count`, `complete`. |
| `engines` | `deterministic` (`ok`, `not_installed`, `failed`, `page_rejected`, `raster_mismatch`), `deterministic_seconds`, `ml` (`provided`, `not_provided`). |
| `gemini` | `status` (`not_needed`, `not_authorized`, `disabled`, `unavailable`, `outside_scope`, `reviewed`), `advisory` (`resolution`, `reason`, `evidence_ids`), `crop_bbox_xyxy`. |

Chaque élément de preuve a la forme `{"id": "deterministic:d0", "provider": "deterministic", "bbox_xyxy": [50, 50, 90, 90]}`. L'avis Gemini est consultatif : l'afficher dans un champ distinct du verdict, sans jamais changer `status` ni `answer`.

Erreurs : `404` si le `document_id` a expiré (le backend garde huit pages en mémoire), `422` si `ml_detections` ne correspond pas à l'image analysée, `413` ou `415` à l'import.

## Démo sans moteur

`POST /api/v1/demo/door-count?scenario=agreement|disputed|missing_ml` renvoie la même forme de réponse (sans `engines`) sur une page synthétique, avec `mode: "synthetic_demo"`. L'étiqueter comme simulation.

## Limites mesurées le 14 novembre

- Le moteur déterministe s'exécute en 1 à 5 secondes par page mais n'a détecté aucune porte sur les deux pages d'essai, qui en contiennent deux chacune. Tant que ce point n'est pas corrigé, l'analyse réelle renvoie `INSUFFICIENT_EVIDENCE`.
- L'adaptateur ML déclare toujours une couverture inconnue. Avec le ML actuel, deux moteurs d'accord donnent donc `coverage_unknown` et pas `SUPPORTED`.
