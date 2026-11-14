# PlanProof × NVIDIA Nemotron

Nemotron examine le recadrage du désaccord entre les moteurs déterministe et ML. PlanProof vérifie les références citées avant d'afficher l'avis.

## Résultats dans Swagger

| Scénario synthétique | Résultat |
| --- | --- |
| Accord | Deux portes confirmées par les deux moteurs. Aucun appel à Nemotron. |
| ML absent | `INSUFFICIENT_EVIDENCE`. Deux détections conservées pour vérification. Aucun appel à Nemotron. |
| Désaccord | Quatre avis corrects sur cinq essais. Les quatre avis ont cité `deterministic:2`. Un avis a été rejeté faute de preuve en faveur du moteur choisi. Les cinq essais sont restés en `NEEDS_REVIEW`. |

Deux des cinq essais ont pris environ 3 à 4 secondes d'après le chronométrage manuel. Les cinq essais utilisent le même recadrage synthétique. La précision sur des plans réels reste à mesurer sur des désaccords annotés.

## Reproduire

Depuis la racine du dépôt, lancer `python backend/check_nemotron.py --serve` et saisir la clé Verda dans le terminal. Dans [Swagger](http://127.0.0.1:8001/docs), lancer `POST /api/v1/demo/door-count` avec `scenario=disputed` et `allow_nemotron_review=true`.

Suite backend locale de 27 tests, dont 26 réussis et 1 ignoré faute des dépendances du moteur déterministe réel. L'intégration couvre le backend et Swagger.
