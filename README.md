# ERP Data-Driven Decision Support System

A Python-based student analytics and decision-support project using feature engineering, preprocessing, K-Means clustering (five project-defined groups), Random Forest evaluation, and an interactive Gradio interface.

## Features
- Academic and survey data integration
- Feature engineering for arrears and credits earned
- Categorical encoding and missing-value handling
- Standardization and K-Means clustering
- Five project-defined performance groups
- Random Forest classification evaluation
- Gradio interface with semester arrears and SGPA visualizations

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Gradio, Jupyter

## Repository Structure
- `app/`: Gradio application
- `src/`: preprocessing and machine-learning modules
- `notebooks/`: Jupyter analysis notebooks
- `docs/`: project notes
- `data/README.md`: instructions for local datasets

## Dataset privacy
The student CSV datasets are intentionally **not included** in this public repository because they contain sensitive student-related information. To run the project, place authorized copies of `student_academic_data_anonymized.csv` and `student_survey_data_anonymized.csv` in the local `data/` folder. Do not upload private student information to GitHub.

## Run locally
1. Create a virtual environment: `python -m venv .venv`
2. Activate it in PowerShell: `.\.venv\Scripts\Activate.ps1`
3. Install dependencies: `pip install -r requirements.txt`
4. Put authorized CSV files in `data/` using the filenames above.
5. Run: `python main.py`

Open the local Gradio URL printed in the terminal.

## Project-defined groups
1. High Achievers
2. Moderate Performers
3. Slow Learners
4. Inconsistent Performers
5. Potential Improvers

These are labels defined by the project workflow, not formal or clinical assessments.
