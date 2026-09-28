# StreamSentinel

AI help for citizens who check urban streams. Built for the OneAquaHealth IEEE Global Hackathon 2026, Track 3 (AI-Supported Assessment).

## The problem

Citizens who report on a stream often send data that is uneven and hard to use. The OneAquaHealth citizen app asks many questions, and some of them are difficult for a non-expert.

StreamSentinel looks at the citizen's photo and pre-fills part of the OneAquaHealth form. It also estimates a risk level for people and for animals, using the photo together with public weather and river flow data. The citizen checks every answer before sending. No alert is sent without a manager's validation.

## How it works

We did not train a big model from scratch. The pipeline combines pretrained models, image processing and one small detector that we fine-tuned. It was built in steps, one notebook per day.

| Day | What it does | Main tools |
|---|---|---|
| J1 | Checks photo quality and segments the scene (water, vegetation, buildings) | SegFormer, OpenCV |
| J2 | Checks that the photo really shows a stream | CLIP |
| J3 | Detects litter and other objects | YOLO11s fine-tuned, OWLv2 |
| J4 | Pre-fills the form with the real OneAquaHealth answer codes | Qwen2.5-VL-3B |
| J5 | Computes the risk for people and animals | Open-Meteo, Hub'Eau |
| J6 | Fixes the errors found in J4 | |

When the model is not sure, it answers NOT_SURE instead of guessing. The question is then left to the citizen.

## Results

We first measured our models on the photos we used to build them. The numbers looked good, but they were too optimistic. So we built independent test sets with photos the models had never seen, and we annotated them by hand.

| Model | Test set | Result |
|---|---|---|
| Form pre-filling (J4) | 21 unseen Wikimedia photos | 71% of answers correct (95% CI 62 to 80%), 61% of questions pre-filled |
| Stream check (J2) | 66 unseen Wikimedia photos | 92% of streams accepted, 79% of off-topic photos rejected |
| Litter detector (J3) | 382 external images | precision 0.77, recall 0.43 |

The litter detector reached a mAP50 of 0.796 on its own validation set of 599 images. On the external set it misses more than half of the litter, so "no litter detected" does not mean the stream is clean. Lowering the detection threshold did not really help (best F1 0.557 against 0.548), so we kept it at 0.25.

Some questions are answered well. Bank type was right 100% of the time and water appearance 89% of the time. Others were close to chance. The model almost never answers the hardest questions, like sewage discharge or vegetation cuts, and we think this is the right behaviour.

## What we changed after the evaluation

For four questions, the accuracy was not better than chance. These are channel shape, vegetation on the left bank, vegetation on the right bank and the overall condition. The AI no longer pre-fills them, and the citizen answers them alone. The model still computes them in the background, so they can be switched back on if a better model passes the test.

We judged the left and right banks together. With so few photos, a difference between the two sides would most likely be noise.

## Risk score

The risk score adds points for each signal the pipeline sees, like foam, abnormal colour, sewage discharge, litter, a dead animal, recent rain or a very high river flow. There is one score for people and one for animals, because dogs are more exposed to algal toxins and people are more exposed to strong currents.

Each rule is linked to the OneAquaHealth Key Indicators Factsheets when they support it. For example, the fecal coliforms and pathogens factsheets explain why sewage discharge is a health risk, and the diatoms factsheet explains why an abnormal water colour can mean a toxic algal bloom. The factsheets do not give thresholds for photo observations. So the points and thresholds are our own choices, and the app says so. Rules with no factsheet behind them are marked as team assumptions.

A low risk gives a simple safety tip. A high risk is sent to the manager's queue and nothing is published until a human validates it.

## The app

The app has three spaces.

1. New observation. The citizen sends a photo and picks the place on a map of the 106 OneAquaHealth research sites. The form then opens in six steps, like the official app. Under each answer, a tag shows if it was suggested by the AI, changed by the citizen or left for the citizen.
2. Analysed examples. Photos analysed in advance, so the app works without the GPU.
3. Manager. The validation queue for high risk observations, with a table that explains every point of the risk score.

Observations can be exported as FHIR R4, following the draft OneAquaHealth implementation guide (hl7-eu/oah). The export also includes the answer codes used by the OneAquaHealth app.

## Limits

The test sets are small, about 20 to 70 photos. The confidence intervals are wide and the results should be read as a first estimate.

The J4 test set was annotated by one person only, so we could not measure the agreement between two annotators.

Wide and calm rivers are sometimes taken for lakes by CLIP, and some open landscapes like the sea pass the stream check.

Hub'Eau river flow data only exists in France. In the other OneAquaHealth cities, the risk uses the weather only.

Face blurring for privacy is planned in the pipeline but not active yet.

The live analysis needs a GPU. We run it in Google Colab and reach it through an ngrok tunnel, which is fine for a demo but not for real use.

## Run the app

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

The Analysed examples and Manager spaces work right away. For the live analysis, run the notebook `API_streamsentinel.ipynb` in Google Colab with a T4 GPU, then put its address in `.streamlit/secrets.toml` as `API_URL`.

The pipeline notebooks (J1 to J5) also run in Colab. J3 needs a Roboflow API key, saved in the Colab secrets as `ROBOFLOW_API_KEY`.

## Data and credits

Test photos come from Wikimedia Commons, with their licences listed in `credits_photos.csv`. The litter data comes from Roboflow Universe (RF100-VL under MIT, Floating Trash Detection under CC BY 4.0). The research sites come from the OneAquaHealth ENORA API.

The risk rules cite the OneAquaHealth Key Indicators of Ecosystem and Biological Health factsheets by Schmeller et al. (2026), doi 10.5281/zenodo.20345207, under CC BY 4.0.

The list of recent changes is in `CHANGELOG.md`.
