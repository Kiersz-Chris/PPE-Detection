"""Runs the Roboflow PPE model on a folder of photos and writes a vest / hard hat report.


Needs the ROBOFLOW_API_KEY environment variable (set it in the same terminal first).
Outputs go in results/report/: summary.txt, per_person.csv, and annotated/ photos.
"""
import csv
import os
import sys
import PPE_Logic

from pathlib import Path

from PIL import Image, ImageDraw
from inference_sdk import InferenceHTTPClient, InferenceConfiguration


MODEL_ID = "ppe-q7u4b/2"
HAT_DRAW_COLORS = {"Blue": "blue", "Red": "red", "White": "white",
                   "Yellow": "yellow", "Uncertain": "orange"}


def draw_results(image_path, records, out_path):
    """Green person box = vest detected, red = no vest detected. Hat and vest get their own boxes."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for r in records:
        color = "lime" if r["vest"] else "red"
        draw.rectangle(r["person_box"], outline=color, width=3)
        label = f"P{r['person']}: {'vest' if r['vest'] else 'no vest'}, {r['hat_color'] or 'no'} hat"
        draw.text((r["person_box"][0] + 4, r["person_box"][1] + 4), label, fill=color)
        if r["vest_box"]:
            draw.rectangle(r["vest_box"], outline="cyan", width=2)
        if r["hat_box"]:
            draw.rectangle(r["hat_box"], outline=HAT_DRAW_COLORS.get(r["hat_color"], "white"), width=2)
    img.save(out_path)


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: py -3.12 ppe_report.py FOLDER [MIN_CONFIDENCE]")
    folder = Path(sys.argv[1])
    min_conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9
    if not folder.is_dir():
        sys.exit(f"Folder not found: {folder}")

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit('Set your key first:  $env:ROBOFLOW_API_KEY = "your_key_here"')

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com", api_key=api_key,
    ).configure(InferenceConfiguration(api_key_transport="header", confidence_threshold=0.3))

    out = Path("results") / "report"
    (out / "annotated").mkdir(parents=True, exist_ok=True)

    photos = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not photos:
        sys.exit(f"No .jpg, .jpeg or .png photos found in {folder}")

    all_records, empty = [], []
    for photo in photos:
        predictions = client.infer(str(photo), model_id=MODEL_ID)["predictions"]
        records = PPE_Logic.analyze(predictions, min_conf)
        print(f"{photo.name}: {len(predictions)} raw detections -> {len(records)} people kept")
        if not records:
            empty.append(photo.name)
        for r in records:
            r["image"] = photo.name
        all_records.extend(records)
        draw_results(photo, records, out / "annotated" / f"{photo.stem}.jpg")

    # One row per person, handy for Excel
    with open(out / "per_person.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "person", "person_confidence", "vest_detected",
                         "vest_confidence", "hat_color", "hat_confidence", "review_notes"])
        for r in all_records:
            writer.writerow([
                r["image"], r["person"], round(r["person_conf"], 2),
                "Yes" if r["vest"] else "No",
                round(r["vest_conf"], 2) if r["vest_conf"] else "",
                r["hat_color"] or "No hat detected",
                round(r["hat_conf"], 2) if r["hat_conf"] else "",
                "; ".join(r["notes"]),
            ])

    report = PPE_Logic.format_report(all_records, len(photos), empty, min_conf)
    (out / "summary.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\nSaved to: {out.resolve()}")


if __name__ == "__main__":
    main()