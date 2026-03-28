# Repository Guidelines

## Project Structure & Module Organization
This repository is organized around a Python speech-emotion workflow. Core model code lives in `models/`, preprocessing and feature generation live in `preprocessing/`, reusable helpers live in `utils/`, inference orchestration lives in `inference/`, and the Gradio entry point is `ui/app.py`. Experiment notebooks are kept in `notebooks/`, runtime settings are centralized in `configs/config.yaml`, and trained weights plus figures are stored in `checkpoints/`. The `data/` path is a symlink to external storage, so treat raw datasets and derived features as environment assets rather than normal source files.

## Build, Test, and Development Commands
Install dependencies with `pip install -r requirements.txt`. Prepare audio data with `python preprocessing/audio_preprocess.py`, then build baseline Mel and MFCC features with `python preprocessing/feature_extract.py`. Use `python ui/app.py` to launch the local Gradio interface for inference and UI checks. Baseline and shared-model training are driven from `notebooks/03_train_emotion.ipynb` and `notebooks/04_train_shared.ipynb`. For a fast code health check before opening a pull request, run `python -m py_compile inference/*.py models/*.py preprocessing/*.py ui/*.py utils/*.py`.

## Coding Style & Naming Conventions
Follow the existing Python style: four-space indentation, `snake_case` for functions, variables, and modules, and `PascalCase` for classes such as `EmotionAwareSpeechPipeline`. Keep new paths, hyperparameters, and dataset options configurable through `configs/config.yaml` instead of hard-coding them into scripts. Match the repository’s current documentation style by keeping comments concise and preserving Chinese user-facing text where it already exists.

## Testing Guidelines
There is no dedicated `tests/` package yet, so validation is currently task-specific. Syntax checking with `python -m py_compile ...` is the minimum requirement. If you change preprocessing, run the relevant script on a small sample; if you change inference or UI behavior, start `python ui/app.py` and verify both model-selection paths still load. If you add automated tests, place them under a new `tests/` directory and name files `test_<module>.py`.

## Commit & Pull Request Guidelines
Recent history uses short, date-stamped commit subjects such as `3.26.18 val_uar && esd`, so keep commit titles compact and focused on the experiment, metric, or configuration changed. In pull requests, explain which modules and config keys were touched, note any dataset or checkpoint assumptions, and include screenshots when changing `ui/app.py` or visualization output. Avoid committing raw dataset contents, and only commit generated checkpoints or figures when they are intentional reproducible artifacts.
