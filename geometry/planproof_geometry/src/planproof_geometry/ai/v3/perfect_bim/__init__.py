"""Library wrapper around the perfect-BIM pipeline modules.

``diagnose_door_placement`` + ``merge_doors_and_windows_bim`` (+
``add_stairs_to_bim`` als Post-Step) produzieren den Gold-Standard-
BIM-Output, den Willi als perfekt bestaetigt hat. Seit ADR-016 liegen
sie als regulaere Module IN diesem Package (vorher: Standalone-Skripte
in ``scripts/``, per ``sys.path``-Hack geladen — byte-identischer
Output beim Move verifiziert). Dieses Package exponiert ihre
``main()``-Flows als Library-Funktionen fuer die Production-Pipeline
(Cloud-Worker).

How it works
------------
Die Wrapper importieren die Module lazy (``_script_loader``), patchen
die Modul-Konstanten (``PDF_PATH`` / ``ANN_PATH`` / ``OUT`` /
``RENDER_MATRIX``) und rufen dann ``main()``. CLI-Nutzung weiterhin
moeglich via ``python -m planproof_geometry.ai.v3.perfect_bim.<modul>``.

Caveats
-------
* Module-level state means concurrent execution would race. The cloud
  worker processes one job at a time so this is safe today; if that
  changes the patching needs to move to a contextmanager that locks.
* The modules ``print`` directly. Cloud-worker logs swallow that for
  now; the bundle output is what matters for the App.
"""
from .door_pipeline import run_door_placement  # noqa: F401
from .full_bim_pipeline import run_full_bim  # noqa: F401
from .stair_pipeline import run_add_stairs  # noqa: F401
