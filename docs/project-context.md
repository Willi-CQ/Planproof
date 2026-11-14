# PlanProof — contexte projet pour l’audit

Projet d’équipe Junction 2026, challenge STOA « Make Blueprints Talk » et intégration Google Gemini. L’utilisateur doit pouvoir poser une question sur un plan et inspecter la région source qui soutient la réponse. Le premier cas intégré porte sur les portes d’une seule page. L’interface de production et la couverture de questions restent à étendre.

## Architecture V3

```mermaid
flowchart TD
    A["Planwise · import et visualisation"] --> B["FastAPI · import PDF, PNG ou JPEG"]
    B --> C["Page RGB canonique · dimensions et empreintes"]
    C --> D["Moteur déterministe Norlan"]
    C --> E["ML Willi · inférence locale"]
    D --> F["Comparateur Python · même page et appariement des portes"]
    E --> F
    F -->|Accord complet| H["Réponse et régions sources"]
    F -->|Désaccord localisé et envoi autorisé| G["Gemini · recadrage original et avis cité"]
    G --> H
    F -->|Couverture inconnue ou moteur indisponible| H
    H --> I["Visionneuse · cadre de page et boîtes"]
```

Python décide du statut. Gemini n’est appelé que pour une divergence localisée et autorisée dans le parcours V3. Il reçoit le recadrage sans annotations, les résumés des moteurs et les identifiants candidats. Il renvoie un avis structuré. Le statut reste `NEEDS_REVIEW` pour la divergence. Plusieurs propositions concordantes ne suffisent pas à prouver que la page a été couverte intégralement.

Le prototype initial de recherche exacte `FD-12` reste disponible via `/api/analyze` et sa démo via `/api/demo`. L’API V3 accepte les résultats des moteurs ou lance leur inférence dans des sous-processus sur le PNG canonique.

## Dépôt et branches

Origine `Willi-CQ/Planproof`. État distant inspecté au début de cette intégration.

| Branche | Révision inspectée | Contenu et rôle |
| --- | --- | --- |
| `main` | `76a3cd7` | Registre de sources publiques et politique de stockage des données. Le registre local `data/manifest.csv` est retiré de cette branche. |
| `deterministic-algorithms` | `de71906` | Bibliothèques de géométrie et CAD existantes sous `geometry/` et spécification 2D sous `docs/deterministic-evidence/`. |
| `ml/trained-baseline-v1` | `82470b0` | Package `ml/`, inférence et entraînement, profils de calibration et poids versionnés via Git LFS. |
| `front` | `a465bba` | Interface Planwise et prototype Node. |
| `norlan-deterministic` | `d4b8c00` | Package `planproof_evidence` et format `planproof-door-result-v1`. |
| `iamd-johan-deterministic-fix` | `24e3702` | Correctif de lecture des segments OpenCV. |
| `iamd-johan-dev` | `b35fe01` puis le commit qui contient ce document | Frontend, orchestration locale, comparaison et Gemini. |

FastAPI sert Planwise à `/planwise/`. Le front affiche la page et ses régions cliquables, les résultats des moteurs et l’avis Gemini. Le prototype Node, son chat et l’amorce JEV sont sur la branche `front`.

Imad et Johan travaillent sur le contrat du backend, le comparateur et Gemini. Willi a publié l’entraînement et l’inférence ML, exécutés durant le projet avec la puissance de calcul Verda selon l’équipe. Le handoff associe Willi et Norlan à la voie déterministe PNG/PDF. Confirmer avec l’équipe qui maintient l’interface.

## Entrées et sorties des moteurs

**ML.** `corrected-3500-global` est le modèle principal pour murs, pièces et ouvertures. `cad-objects-3000-global` se limite aux portes, fenêtres et escaliers. Le modèle principal utilise une vue globale à 1024 avec son profil de calibration. Le package exige Python 3.11 ou 3.12, distinct de l’environnement backend si nécessaire. Le checkpoint principal vérifié mesure 97 941 718 octets et a pour SHA-256 `c1861511904636f16433d5a0ec7b925c53a323c975a10df7e1886666a6e1bc54`. Le rapport interne décrit 2 051 dessins dans le test projet, IoU murs 0,713, mAP50 objets 0,587 et F1 pièces 0,739. Ces métriques évaluent des composants et ne démontrent pas l’exactitude des réponses sur STOA. [Livraison ML](https://github.com/Willi-CQ/Planproof/tree/ml/trained-baseline-v1/models/baselines/v1)

**Déterministe.** `planproof_cad` convertit un DXF en scène et `api.takeoff` calcule des quantités. Le worker Norlan traite les pixels du PNG canonique sans données ML. Le backend vérifie le SHA-256 du PNG, celui des pixels et les dimensions avant de relier le résultat à la page affichée. Le moteur renvoie ses propositions et son statut via `to_door_result_v1`. L’exploitation de primitives PDF natives et l’OCR restent à évaluer dans ce parcours. [Contrats déterministes](https://github.com/Willi-CQ/Planproof/blob/deterministic-algorithms/docs/deterministic-evidence/CONTRACTS.md)

## Surface de l’API

| Route | Raccordement |
| --- | --- |
| `POST /api/documents` | Multipart `file` et `page` facultatif à partir de 1. PDF, PNG ou JPEG. Renvoie un `document_id`, `image_url` et les dimensions du raster. |
| `GET /api/documents/{document_id}/image` | PNG RGB canonique pour la visionneuse et l’inférence. |
| `GET /api/v1/documents/{document_id}/frame` | Index de page à partir de 0, SHA-256 du fichier et des pixels RGB, largeur, hauteur et convention des coordonnées. |
| `POST /api/v1/door-count` | JSON avec `document_id`, `deterministic`, `ml_detections` brut ou `ml` normalisé et `allow_gemini_review` facultatif. |
| `POST /api/v1/documents/{document_id}/analyze-doors` | Exécute les moteurs configurés. `allow_gemini_review` vaut false par défaut. |
| `GET /api/v1/engines` | Indique quels moteurs sont configurés. |
| `POST /api/v1/demo/door-count` | Démo synthétique nommée `agreement`, `disputed` ou `missing_ml`. |

Les résultats normalisés suivent `planproof-door-result-v1`. Un moteur renseigne sa version, son statut, son indépendance, la couverture et une liste d’assemblages de porte avec ID et boîte. Les boîtes suivent `[x0, y0, x1, y1]` dans le raster affiché, origine en haut à gauche. Les coordonnées sont en pixels. Le `document_id` API est un identifiant de session. Dans le JSON natif de Willi il faut omettre `--document-id` à l’inférence afin que le champ contienne le hash des pixels RGB. L’adaptateur valide ce hash, les dimensions, la page et les boîtes avant de lier les détections à la session. La couverture brute du ML reste inconnue.

La réponse porte `status`, `answer` entier ou null, `reasons`, `source`, `image_url`, les paires et leurs boîtes, les boîtes contestées, les propositions non vérifiées et l’avis `gemini`. Le frontend doit afficher une proposition isolée comme non vérifiée. La politique actuelle demande un appariement univoque avec IoU ≥ 0,5 et une couverture complète déclarée par les deux moteurs pour publier un nombre exact. Ce seuil n’a pas encore été calibré sur les plans STOA.

## Exécution et vérification

Depuis la racine du dépôt avec Python et un environnement dédié au backend.

```powershell
python -m pip install -r backend/requirements.txt
python -m unittest discover -s backend -v
python -m uvicorn app:app --app-dir backend
```

Pour un essai réel Gemini, `python backend/check_gemini.py --serve` invite à saisir la clé du projet dans le terminal. Le script envoie un seul recadrage de la démo synthétique et lance le serveur local sur le port 8001 si l’appel réussit. Le 14 novembre, l’équipe a observé une réponse réelle `SUPPORTS_DETERMINISTIC` citant `deterministic:2`, avec statut PlanProof `NEEDS_REVIEW`. Le résultat des moteurs sur cette page était simulé. Le guide explique comment raccorder les sorties du ML et de Gemini. [Commandes de démo et intégration](gemini-arbitration.md)

Le backend a passé 13 tests automatisés. Ils couvrent l’import, la correspondance avec l’image affichée, les divergences de position, les doublons, les entrées invalides et les réponses Gemini simulées. Le transport simulé vérifie aussi la sérialisation effective du PNG et du schéma par le SDK Google.

Un essai séparé a exécuté le vrai modèle `step-3500-c18615119046` avec Torch CPU et le profil de calibration du bundle, sur le PNG de la démo synthétique. La commande a pris 36,4 secondes sur cette machine, démarrage et chargement du modèle inclus. Elle a produit zéro instance sur ce dessin. L’API a accepté le JSON natif lié au raster et a renvoyé `INSUFFICIENT_EVIDENCE` avec une réponse nulle. Cet essai valide le raccordement ML à l’API, et ne constitue ni une mesure de précision ni un test des deux moteurs réels.

Le parcours navigateur a testé l’exemple simulé avec sélection de sa zone puis l’import du même PNG synthétique et l’exécution réelle des deux moteurs. Les deux moteurs ont proposé zéro détection, la couverture ML est inconnue et Planwise affiche des preuves insuffisantes. Gemini n’a pas été appelé pour cette exécution. L’évaluation des détections et la revue Gemini de divergences sur des plans réels restent à réaliser. Le correctif OpenCV a passé 66 tests du moteur de Norlan, avec 2 tests ignorés.

## Priorités de l’audit Claude Code

1. **Valider la provenance.** Les données envoyées par l’API supposent deux producteurs locaux de confiance. Une empreinte valide lie l’image mais n’authentifie pas un résultat fourni par un client. Avant d’exposer publiquement le service, établir une frontière de confiance côté serveur et contrôler qui peut demander un appel Gemini.
2. **Vérifier la cohérence des boîtes.** Tester références, orientation, page et correspondance entre raster affiché et image analysée sur un PDF vectoriel et un scan autorisés. Analyser à part les erreurs de détection, d’appariement et de couverture.
3. **Évaluer les deux moteurs.** Garder leurs versions et tester le même PNG annoté. Mesurer les classes supportées et les détections manquées sur des pages réelles.
4. **Évaluer la visionneuse.** Vérifier les cadres sur des plans contenant des détections positives. Les résultats doivent rester inspectables lors d’une indisponibilité de Gemini ou de l’un des moteurs.
5. **Contrôler le lancement et le déploiement.** Le backend est local, conserve huit pages en mémoire et perd son état au redémarrage. Les traitements bloquants et les ressources d’upload nécessitent des limites adaptées avant une exposition multiutilisateur. Les sources et poids doivent suivre le registre du projet. Les clés, documents sensibles et fichiers générés restent hors du commit.

Le premier test complet sur un exemple autorisé doit reproduire une question sur les portes, deux sorties authentiques, une comparaison sur la même page et un clic vers une région qui permet de juger la réponse.
