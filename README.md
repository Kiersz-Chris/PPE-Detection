# PPE Detection Report

This project checks site photos for people, safety vests, and hard hats. It uses a Roboflow computer vision model to find them, then my Python code decides which vest and hat belong to which person and writes a report.

## How it works

1. Each photo is sent to a Roboflow detection model.
2. The code filters out low-confidence detections.
3. Vests and hats are matched to people based on where their boxes sit.
4. Unclear cases (for example, a hat detected as two colors) are flagged for human review.
5. The program saves annotated photos, a spreadsheet, and a text summary.

## Setup

Requires Python 3.12

```
git clone https://github.com/YOUR_USERNAME/ppe-detection.git
cd ppe-detection
pip install -r requirements.txt
```

### API key

You need a free Roboflow account. Copy your Private API Key from Settings > API Keys in Roboflow, then set it in your terminal:

Windows PowerShell:
$env:ROBOFLOW_API_KEY = "your_key_here"

The key only lasts for that terminal window.

## Run it

python PPE_Report.py PATH_TO_PHOTO_FOLDER

To try it on the included sample photos:
python PPE_Report.py test_images

Photos must be .jpg, .jpeg, or .png.

## Output

Results are saved in the `results/report/` folder:

- `summary.txt`: totals and a breakdown by hard hat color
- `per_person.csv`: one row per person, opens in Excel
- `annotated/`: photos with boxes and labels drawn on

Example outputs are in `samples/output/`.

## Limitations

- "No vest detected" does not prove someone is not wearing a vest. It may be hidden, blurry, or far away.
- Matching is based on position, so it can be wrong when workers overlap.
- Results are an automated first pass and need human verification.

## Possible next steps

- Add a simple web interface
- Detect more PPE such as gloves and safety glasses
- Measure accuracy against hand-labeled photos