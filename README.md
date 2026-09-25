# Predicting stroke 2026-27

**Setup (eenmalig):**

1. Accepteer de GitHub-uitnodiging en open de repo in een Codespace (Code → Codespaces → Create codespace on main).
2. In de terminal: `pip install -r requirements.txt`
3. Join de Kaggle-competitie en accepteer de team-uitnodiging.

**Werkwijze:** we werken allemaal op `main`, ieder in een eigen notebook.

* Voor je begint: `git pull`
* Na je werk sla je je notebook op (Ctrl + S) en voer je deze 3 commando's uit in de terminal. Vervang `03_knn.ipynb` door de naam van jouw notebook:

```bash
  git add notebooks/03_knn.ipynb
  git commit -m "KNN: hyperparameter tuning toegevoegd"
  git push
```

  - `git add` selecteert jouw notebook.
  - `git commit` slaat een versie op met een korte beschrijving van wat je hebt gedaan.
  - `git push` uploadt die versie naar GitHub, zodat de rest hem kan zien.
* Voeg alleen je eigen notebook toe, niet `git add .`

**Data inladen:** zet deze code in de eerste codecel van je notebook. Je krijgt dan direct de opgeschoonde data zoals beschreven in `01_eda.ipynb`:

```python
import sys

sys.path.append("..")  # zodat de map src gevonden wordt

from src.preprocessing import NUMERIC_COLS, StrokeData

data = StrokeData()
X, y = data.get_xy()       # features en target van de trainset
X_test = data.get_test()