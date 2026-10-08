# src/models — deep learning model slot

The pipeline has two swappable model slots, defined as abstract classes in `base.py`:

| Slot (`configs/models.yaml` key) | Interface | `predict` input → output |
|---|---|---|
| `detector` | `Detector` | list of frames → list (per frame) of boxes `{x, y, width, height, score}` |
| `impact_classifier` | `ImpactClassifier` | list of crop sequences → list of impact probabilities |

Both also expose a `name` property and `load(weights_path)`. Nothing here assumes a framework.

## Adding a model

1. Create a module, e.g. `src/models/my_detector.py`:

   ```python
   from src.models.base import Detector
   from src.models.registry import register

   @register("my_detector")
   class MyDetector(Detector):
       @property
       def name(self) -> str:
           return "my_detector"

       def load(self, weights_path: str) -> None:
           ...  # load weights with whatever framework you like

       def predict(self, frames):
           ...  # return one list of boxes per frame
   ```

2. Import the module somewhere that runs before lookup (e.g. add it to `src/models/__init__.py`) so the decorator executes.
3. Point the slot at it in `configs/models.yaml`:

   ```yaml
   detector:
     name: my_detector
     weights_path: weights/my_detector.pt   # weights are gitignored
   ```

4. Load it with `get_model_from_config("detector")`.

Framework dependencies go in a separate requirements file, not `requirements-agent.txt`.
