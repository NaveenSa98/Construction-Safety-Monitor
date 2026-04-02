# Dataset Documentation

## How the Dataset Was Built

The training dataset was created by taking three existing public datasets and extending them with a fourth specialised source to cover all required PPE classes. All four sources use YOLO `.txt` annotation format. Class names were remapped to a shared master list and duplicate images were removed before merging.

- **Final size: 4000 images · 12 classes · 70 / 20 / 10 split**

---

## Sources

| 1 | PPE Detection | Roboflow — tanish-y7iqo/ppe-detection_data-adcya v3 | 2,801 | Hardhat, Vest, Boots, Gloves, Goggles, Mask |
| 2 | Construction PPE | Ultralytics — construction-ppe.yaml | 1,416 | Person, Hardhat, Vest, Boots, Gloves, Goggles + NO- variants |
| 3 | Construction Site Safety | Roboflow — roboflow-universe-projects/construction-site-safety | ~2,000* | Person, Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest |
| 4 | Safety PPE | Hugging Face — safety-jmser/safety_ppe | 6,629 | Full PPE set including Safety Harness, Boots, Gloves, Goggles + NO- variants |

*PPE-relevant subset only. Source 3 originally contains 25 classes; the 18 non-PPE classes (vehicles, machinery, etc.) were dropped.

---

## My Additions

Source 3 (Roboflow Construction Site Safety) was used as the **base dataset**. The following three sources were added to extend coverage across all required PPE classes:

- **Source 1** was added to introduce Safety Boots, Safety Gloves, and Safety Goggles annotations, which are absent in the base dataset.
- **Source 2** was added for its negative classes (NO-Hardhat, NO-Safety Vest, NO-Safety Boots, NO-Safety Gloves, NO-Safety Goggles), which are essential for violation detection.
- **Source 4** was added as the sole source covering Safety Harness and NO-Safety Harness, and to further strengthen Boot and Glove class representation.

All class names were normalised to a unified 12-class scheme using a custom merging script (`build_dataset.py`). Duplicate images were removed using SHA-256 exact hashing and perceptual hashing (pHash).

---

## Class List

| 0 | Person                      | 6 | NO-Hardhat |
| 1 | Hardhat                     | 7 | NO-Safety Vest |
| 2 | Safety Vest                 | 8 | NO-Safety Boots |
| 3 | Safety Boots                | 9 | NO-Safety Gloves |
| 4 | Safety Gloves               | 10 | NO-Safety Goggles |
| 5 | Safety Goggles

---

## Known Limitations

- **Lighting coverage** is limited to daylight and indoor construction conditions. Nighttime or low-light performance is not guaranteed.
- **Class imbalance** exists — core classes (Person, Hardhat, Safety Vest) have significantly more annotations than Goggles and Harness.
