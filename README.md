# Biophilic Sound Transform

A Python-based toolkit for analyzing and transforming ambient office soundscapes into biophilic auditory experiences.

##  Overview
This project explores how to leverage real-world office sound data to create nature-inspired sound transformations. It includes:
- **Preprocessing**: Audio segmentation, normalization, and cleaning.
- **Feature Extraction**: Temporal, spectral, and envelope-based features for sound classification.
- **Three-Tier Classification**: Rule-based sound type assignment for biophilic design mapping.
- **Biophilic Transformations**: Modular transformations to create natural sound overlays or modifications.
- **Evaluation**: Tools and methods to assess perceptual and ecological validity.

##  Project Structure
- `data/`: Raw and processed audio data.
- `biophilic_transform/`: Core Python modules (preprocessing, feature extraction, classification, transformation).
- `notebooks/`: Jupyter notebooks for exploration and visualizations.
- `scripts/`: CLI tools for batch processing.
- `features/`: Extracted feature data.
- `rules/`: Rule mapping JSON files.
- `tests/`: Unit tests for reproducibility.

## Setup
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

