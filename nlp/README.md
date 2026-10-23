# Nova NLP: rule-based vs learned intent classification

Question: **does a trained text classifier understand friendly chat messages better than hand-written keyword rules?**

## Setup

- **18 intents** (greeting, thanks, goodbye, how_are_you, sad, stressed, anxious, angry, lonely, bored, happy, music, joke, game, study_help, career_help, sleep_help, breathing)
- **Train:** 288 messages (16 per intent). **Test:** 126 messages (7 per intent), written as *different phrasings* from the training set so keyword overlap can't inflate the score.
- **Out-of-scope:** 30 off-topic messages ("how do vaccines work") to test whether the model knows what it doesn't know.
- **Baseline:** the same keyword patterns and priority order Nova's rule engine uses (`baseline.py`).
- **Models:** TF-IDF + logistic regression, with word n-grams only vs word + character n-grams.
- **Model selection:** 5-fold cross-validation on the training set only. The test set was used once at the end, so it is not tuned on.

## Results (held-out test set, 126 messages)

| Model | Accuracy | Macro F1 |
|---|---|---|
| Rule-based baseline | 0.31 | 0.42 |
| TF-IDF words + LR | 0.69 | 0.68 |
| **TF-IDF words + chars + LR (selected)** | **0.76** | **0.77** |

The rules fail to match anything for 86 of 126 messages (68%). The learned model generalises to wordings it never saw ("my blood is boiling", "i toss and turn all night").

Figures: `results/confusion_matrix.png`, `results/per_intent_f1.png`, `results/learning_curve.png`. All errors are in `results/errors.csv`.

## Knowing when not to answer

A confidence threshold decides when the model should act. At 0.25:

- it answers 79% of in-scope messages, and 89% of those answers are correct
- it rejects 70% of out-of-scope messages

The browser version uses this: high confidence acts directly, medium confidence asks "did you mean...?", low confidence admits it didn't understand.

## Honest limitations

- **Small, hand-written dataset.** Same author for train and test, so real users will phrase things differently and the real score will probably be lower. The learning curve is still rising at 16 examples per intent, so more data would help.
- **Weak intents:** `happy` (F1 0.24) and `career_help` are the hardest. "i aced my test" has almost no word overlap with the training examples.
- **Out-of-scope rejection is imperfect (70%).** A 30% leak is why Nova confirms before acting on medium-confidence guesses.
- The baseline was ported from Nova's rules, so it reflects the keyword approach, not the best possible rule system.

## Reproduce

```bash
pip install -r requirements.txt
python train.py
```

This retrains, regenerates `results/` and exports `../model.js` for the browser. The JavaScript implementation was checked against scikit-learn on 156 messages: maximum probability difference 4e-4.

## Files

| File | What |
|---|---|
| `data/train.csv`, `data/test.csv`, `data/out_of_scope.csv` | Labelled messages |
| `baseline.py` | Keyword-rule baseline |
| `train.py` | Training, evaluation, plots, export |
| `results/` | Metrics, figures, error list |
