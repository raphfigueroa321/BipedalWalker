# BipedalWalker

This repository trains and evaluates PPO agents on the Gymnasium Box2D
BipedalWalker-v3 environment. It includes training recipes, light reward
wrappers (hull-angle and leg-contact penalties), and scripts to record GIFs,
visualize runs, and compute final statistics from TensorBoard logs. This current model can "solve" the normal version and reach 300 points in 1600 time steps.

**Requirements**
- Python 3.8+ and packages listed in `requirements.txt`.

Install:

```bash
pip install -r requirements.txt
```

**Quick Start**
- Train a baseline agent:

```bash
python baseline.py
```

- Train experiments with hull/leg penalties:

```bash
python main.py
```

Models will be saved in respective directories.

Results folder: The `results/` directory holds the best runs and artifacts used for quick evaluation. Typical structure:
- `results/best_model/` — contains the selected best run with `walker_model.zip` and (optionally) `vec_normalize.pkl` for loading and evaluation.
- `results/gifs/` — generated GIFs and `gif_metadata.json`.
- `results/logs/` — aggregated per-seed log exports used for plotting or quick inspection.

Use the `results/best_model` path when running `visualize.py` or `record_gif.py` to quickly load and render the top-performing model.

- Record GIF(s) from a saved model (provide the folder containing `walker_model.zip`):

```bash
python record_gif.py models/best_model --output-root gifs --episodes 1 --max-steps 2000 --fps 30
```

- Visualize a saved model interactively (set `MODEL_PATH` in `visualize.py`):

```bash
python visualize.py
```

- Compute final statistics (mean/standard deviation of reward across seeds) from TensorBoard logs:

```bash
python calculatestats.py
```

