# Dataset Documentation

## 1. Base Dataset — Roboflow Public Dataset

### Source Information

- **Dataset Name**      PPE Detection Computer Vision Dataset
- **Platform**          Roboflow Universe
- **URL**               <https://universe.roboflow.com/roboflow-universe-projects/ppe-detection-new-model>
- **License**           CC BY 4.0
- **Format**            YOLOv8 (`.txt` annotations + images)  
- **Total Images**      2801 images
- **Split**             Train(2605) / Validation(114) / Test(82)
- **Annotation Format** Bounding box — YOLO `.txt` format

---

### Classes Covered

`helmet` / `hard-hat`| `vest` / `hi-vis` | `person` / `worker` | `no-helmet` | `no-vest`

---

## 2. Custom Dataset

### Custom Source 1 — Safety Goggles

- **Dataset:** goggles-ppe
- **Total Images:** 179
- **Classes Used:** Goggles → remapped to Safety Goggles (index 9)
- **Addresses Rule:** R7 — Protective Eyewear Required

---

### Custom Source 2 — Safety_PPE Dataset

- **Dataset:** safety-jmser/safety_ppe
- **Total Images:** 6,629
- **Split:** Train 5,001 / Valid 1,328 / Test 300

- **Classes Used:**
Safety Gloves, Safety Goggles, Hardhat, NO-Safety Gloves, NO-Safety Goggles, NO-Safety Harness, NO-Hardhat,
NO-Safety Boots, Person, Safety Harness, Safety Boots

---

### Custom Source 3 — Safety Gloves

- **Dataset:** safety-gloves-xbnf8
- **Total Images:** 3,373
- **Classes Used:**
  - Gloves    → Safety Gloves    (index 8)
  - NO-Gloves → NO-Safety Gloves (index 10)
- **Addresses Rule:** R6 — Protective Gloves Required

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
