"""Turns raw model detections into per-person PPE results and a text report.

Class codes for the Roboflow model ppe-q7u4b/2:
    0 = person, 1 = vest, 2 = blue hat, 3 = red hat, 4 = white hat, 5 = yellow hat
=
"""

HAT_COLORS = {"2": "Blue", "3": "Red", "4": "White", "5": "Yellow"}
HAT_ORDER = ["Blue", "Red", "White", "Yellow", "Uncertain", None]  # None = no hat found


# ---------- geometry helpers ----------

def to_corners(p):
    """Takes the center, height, and width, and changes it to the corners"""
    return (p["x"] - p["width"] / 2, p["y"] - p["height"] / 2,
            p["x"] + p["width"] / 2, p["y"] + p["height"] / 2)


def area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def overlap_area(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0, w) * max(0, h)


def iou(a, b):
    """percent of total area overlapping"""
    inter = overlap_area(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union else 0


# ---------- matching rules (the heuristics) ----------

def vest_score(vest_box, person_box):
    """returns the fraction that the vest box is inside a person box"""
    return overlap_area(vest_box, person_box) / area(vest_box) if area(vest_box) else 0


def hat_score(hat_box, person_box):
    """How well a hat fits a person's head position. This is calculated to be the top of the persons box"""
    left, top, right, bottom = person_box
    w, h = right - left, bottom - top
    if w <= 0 or h <= 0:
        return None
    cx, cy = (hat_box[0] + hat_box[2]) / 2, (hat_box[1] + hat_box[3]) / 2
    # Hat must be roughly above the person horizontally...
    if not (left <= cx <= right):
        return None
    # ...and near the top of the person's box (a little above or below the top edge).
    if not (top - 0.35 * h <= cy <= top + 0.30 * h):
        return None
    dx = abs(cx - (left + right) / 2) / w
    dy = abs(cy - top) / h
    return 1 - min(1, dx + dy)


def best_person(people, box, obj_type, minimum):
    best = None
    best_score = None

    for person in people:
        if obj_type == "vest":
            score = vest_score(box, person["box"])
        else:
            score = hat_score(box, person["box"])

        if score is None or score < minimum:
            continue

        if best is None or score > best_score:
            best = person
            best_score = score

    return best


# ---------- main analysis ----------

def analyze(predictions, min_conf=0.9):
    """predictions: Roboflow 'predictions' list for ONE photo.
    Returns one record (dict) per person found."""
    preds = []
    for p in predictions:
        if p["confidence"] >= min_conf:
            preds.append(p)

    people = []
    for p in preds:
        if str(p["class"]) == "0":
            person = {
                "box": to_corners(p),
                "conf": p["confidence"],
                "vests": [],
                "hats": []
            }
            people.append(person)
    people.sort(key=lambda person: person["box"][0])  # number people left to right

    for p in preds:
        cls = str(p["class"])
        box = to_corners(p)
        if cls == "1":
            # A vest belongs to the person whose box contains at least half of it.
            owner = best_person(people, box, "vest", 0.5)
            if owner:
                owner["vests"].append((p["confidence"], box))
        elif cls in HAT_COLORS:
            owner = best_person(people, box, "hat", 0.0)
            if owner:
                owner["hats"].append((p["confidence"], box, HAT_COLORS[cls]))

    records = []
    for i, person in enumerate(people, start=1):
        notes = []
        vest = None
        if person["vests"]:
            vest = person["vests"][0]

            for v in person["vests"]:
                if v[0] > vest[0]:
                    vest = v
        hats = person["hats"].copy()
        for i in range(len(hats)):
            for j in range(i + 1, len(hats)):
                if hats[j][0] > hats[i][0]:
                    hats[i], hats[j] = hats[j], hats[i]

        hat_conf = hat_box = hat_color = None
        if hats:
            hat_conf, hat_box, hat_color = hats[0]
            # The same hat detected as two different colors then  don't guess.
            if len(hats) > 1 and hats[1][2] != hat_color and iou(hats[0][1], hats[1][1]) > 0.5:
                notes.append(f"hat detected as both {hat_color.lower()} and {hats[1][2].lower()}")
                hat_color = "Uncertain"
        if person["conf"] < 0.7:
            notes.append("low-confidence person detection")

        records.append({
            "person": i,
            "person_conf": person["conf"],
            "person_box": person["box"],
            "vest": vest is not None,
            "vest_conf": vest[0] if vest else None,
            "vest_box": vest[1] if vest else None,
            "hat_color": hat_color,
            "hat_conf": hat_conf,
            "hat_box": hat_box,
            "notes": notes,
        })
    return records


# ---------- summary and report text ----------


def summarize(records):
    """Returns (total people, people with vest, [(hat color, [people, with vest]), ...])."""
    by_hat = {}

    total = 0
    with_vest = 0

    for r in records:
        total += 1

        if r["vest"]:
            with_vest += 1

        color = r["hat_color"]

        if color not in by_hat:
            by_hat[color] = [0, 0]

        by_hat[color][0] += 1

        if r["vest"]:
            by_hat[color][1] += 1

    ordered = []

    for color in HAT_ORDER:
        if color in by_hat:
            ordered.append((color, by_hat[color]))

    return total, with_vest, ordered


def pct(n, d):
    return f"{100 * n / d:.0f}%" if d else "n/a"


def format_report(records, photo_count, empty_photos, min_conf):
    total, with_vest, by_hat = summarize(records)
    lines = [
        "MINE PPE REPORT (automated visual assessment - human verification required)",
        "=" * 75,
        f"Photos analyzed: {photo_count}",
        f"Confidence threshold: {min_conf:.2f}",
        "",
        "OVERALL",
        f"  People detected:      {total}",
        f"  Vest detected:        {with_vest} ({pct(with_vest, total)})",
        f"  No vest detected:     {total - with_vest} ({pct(total - with_vest, total)})",
        "",
        "BY HARD HAT COLOR",
        f"  {'Hat color':<20}{'People':>7}   {'Vest detected':<16}{'No vest detected':<16}",
    ]
    for color, (n, v) in by_hat:
        label = (color or "No hat detected") + ("*" if n < 5 else "")
        lines.append(f"  {label:<20}{n:>7}   {f'{v} ({pct(v, n)})':<16}{f'{n - v} ({pct(n - v, n)})':<16}")
    lines += [
        "",
        "NOTES",
        "  *This is not 100 percent effective and should be checked by",
        " a human. Hat percentages are for detecting compliance among ",
        " different members of the working group."
    ]

    review = [(r["image"], r["person"], "; ".join(r["notes"])) for r in records if r["notes"]]
    if review or empty_photos:
        lines += ["", "ITEMS FOR HUMAN REVIEW"]
        for image, person, note in review:
            lines.append(f"  {image}, person {person}: {note}")
        for name in empty_photos:
            lines.append(f"  {name}: no people detected")
    return "\n".join(lines)