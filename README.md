# Süper Lig Match Outcome Prediction

A machine learning and Streamlit project for predicting the outcome of Turkish Süper Lig matches using historical match results and team-level performance statistics.

The application estimates the probability of:

- Home win
- Draw
- Away win

> **Disclaimer:** This project was developed for educational and portfolio purposes. Its predictions should not be used for betting or financial decisions.

## Project Overview

The project uses historical Süper Lig match data covering multiple seasons.

The workflow includes:

- Combining season-level match datasets
- Cleaning and preparing match records
- Encoding team identities
- Creating team performance statistics
- Training multiple classification models
- Combining selected models with soft voting
- Saving the trained model and preprocessing artifacts
- Presenting predictions through a Streamlit interface

## Prediction Type

This project predicts the **match outcome**, not the exact final score.

The target classes are:

- `0` — Draw
- `1` — Home team wins
- `2` — Away team wins

## Application Features

The Streamlit application allows users to:

- Select a home team
- Select an away team
- Compare team performance statistics
- Generate a match-outcome prediction
- View home-win, draw, and away-win probabilities
- See a confidence-oriented interpretation of the result

## Model Architecture

The current application uses a soft-voting ensemble containing:

- Random Forest Classifier
- Support Vector Machine
- Logistic Regression

The individual models produce class probabilities, which are combined by a `VotingClassifier`.

## Model Inputs

The trained model uses features such as:

- Home-team average points
- Away-team average points
- Home-team attacking average
- Away-team defensive average
- Difference in team-strength values
- Encoded home-team identity
- Encoded away-team identity

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- Jupyter Notebook

## Repository Structure

```text
super-lig-match-outcome-prediction/
├── app.py
├── train_ensemble_model.py
├── data_preparation_and_model_comparison.ipynb
├── egitim_verisi_final.csv
├── final_voting_model.pkl
├── 1990_91_sezon_maclari.csv
├── ...
├── 2024_25_sezon_maclari.csv
├── requirements.txt
├── .gitignore
└── README.md
```

The season-level CSV files contain the historical match records used during data preparation.

## Installation

Clone the repository:

```bash
git clone https://github.com/betul-bostan/super-lig-match-outcome-prediction.git
cd super-lig-match-outcome-prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Training the Model

Run the model-training script:

```bash
python train_ensemble_model.py
```

The script reads:

```text
egitim_verisi_final.csv
```

and generates:

```text
final_voting_model.pkl
```

The saved bundle contains:

- Trained voting model
- StandardScaler
- Team-strength statistics
- Team-name and team-ID mappings

## Running the Application

Run:

```bash
streamlit run app.py
```

The application will open in your browser.

## Experimental Results

Earlier experiments compared multiple classification approaches.

The recorded results included approximately:

| Model | Accuracy |
|---|---:|
| Random Forest | 52.94% |
| SVM | 52.84% |
| Logistic Regression | 52.65% |
| Artificial Neural Network | 49.04% |

These figures should be treated as experimental results from the current notebook workflow.

They should be verified again using a reproducible, time-aware train/test split before being interpreted as final performance metrics.

## Important Evaluation Note

Football-match prediction is a high-variance problem.

The available features do not include several factors that can affect a real match, such as:

- Player injuries and suspensions
- Starting lineups
- Recent match-by-match form
- Manager changes
- Home-stadium conditions
- Weather
- Referee tendencies
- Match importance
- Transfer-period changes

The current model therefore represents an educational baseline rather than a production-grade forecasting system.

## Data Leakage Risk

The current team-strength statistics are calculated using the combined historical dataset.

For stricter evaluation, team statistics should be computed using only matches that occurred before each predicted match.

A future version should use chronological feature generation and season-based or rolling-window validation.

## Limitations

- The application predicts outcomes rather than exact scores.
- Historical team IDs may not fully represent changing squads and club strength.
- The model does not currently use true recent-form windows.
- Some external match factors are unavailable.
- Team statistics may introduce look-ahead bias during evaluation.
- Accuracy alone is not sufficient for evaluating an imbalanced three-class problem.
- Dataset sources and licenses should be documented.

## Future Improvements

- Generate team features chronologically
- Use rolling recent-form statistics
- Apply season-based train/test splitting
- Report confusion matrix and per-class F1 scores
- Compare against simple baselines
- Add class-distribution analysis
- Include Elo ratings
- Add injury, lineup, referee, and weather data
- Organize raw and processed data into separate folders
- Add automated tests
- Add application screenshots
- Deploy the Streamlit application
- Document dataset sources and licenses

## Author

**Betül Bostan**

- [GitHub](https://github.com/betul-bostan)
- [LinkedIn](https://www.linkedin.com/in/bet%C3%BCl-bostan-2105942b2/)
