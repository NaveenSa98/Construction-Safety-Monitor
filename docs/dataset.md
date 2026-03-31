# Dataset Documentation

## Dataset

### Source 1 - Safety dataset

- **Dataset Name**      PPE Detection Computer Vision Dataset
- **Format**            YOLOv8 (`.txt` annotations + images)  
- **Total Images**      2801 images
- **Split**             Train(2605) / Validation(114) / Test(82)
- **Annotation Format** Bounding box — YOLO `.txt` format

**Classes Covered:**

- helmet / hard-hat, vest / hi-vis, person / worker, no-helmet, no-vest

---

### Source 2 — Safety Goggles

- **Dataset:** goggles-ppe
- **Total Images:** 179
- **Classes Covered:** Safety Goggles
- **Addresses Rule:** R7 — Protective Eyewear Required
- **Annotation Format** Bounding box — YOLO

---

### Source 3 — Safety_PPE Dataset

- **Dataset:** safety-jmser/safety_ppe
- **Total Images:** 6,629
- **Split:** Train 5,001 / Valid 1,328 / Test 300
- **Annotation Format** Bounding box — YOLO
- **Classes Covered:**
Safety Gloves, Safety Goggles, Hardhat, NO-Safety Gloves, NO-Safety Goggles, NO-Safety Harness, NO-Hardhat,
NO-Safety Boots, Person, Safety Harness, Safety Boots

---

### Source 4 — Safety Gloves

- **Dataset:** safety-gloves-xbnf8
- **Total Images:** 3,373
- **Classes Covered:**
  - Safety Gloves
  - NO-Safety Gloves
- **Addresses Rule:** R6 — Protective Gloves Required
- **Annotation Format** Bounding box — YOLO

## Known Data Limitation

- **Safety Harness (R5):** No suitable public dataset with sufficient
  harness-specific annotations was identified within the project
  timeline. Rule R5 is fully defined in the safety rules specification
  but is marked as **data-limited** in the current model. The compliance
  logic layer handles this gracefully by marking harness evaluation as
  NOT EVALUABLE when no harness class detection is available.

- **Class Distribution:**

  |----------------|-------|
  | Person         | [X]   |
  | Hardhat        | [X]   |
  | NO-Hardhat     | [X]   |
  | Safety Vest    | [X]   |
  | Gloves         | [X]   |
  | NO-Gloves      | [X]   |
  | Goggles        | [X]   |
  | Harness        | [X]   |
  | NO-Safety Vest | [X]   |
  | Boots          | [X]   |
  | NO-Boot        | [X]   |

---

## Combined Dataset Summary

- Base Dataset (Construction Site Safety) | 7,000 | Person, Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest |
- Goggles Dataset | 179 | Safety Goggles
- Footwear Dataset | 7,983 | Safety Boots, NO-Safety Boots |
- Gloves Dataset | 3,373 | Safety Gloves, NO-Safety Gloves |
- **Total** | **18,000+** | **10 active classes** |

---

**Note:**

- Custom images collected from publicly available web sources. So limited representation of nighttime or artificial lighting scenarios

---
