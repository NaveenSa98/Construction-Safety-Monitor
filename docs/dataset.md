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

- **Collection Method:** Manual collection via Google Images search.
- **Number of Custom Images:** 150 images collected.
- **Scenes Covered:** Outdoor construction sites, indoor warehouses, PPE vialations workers, daylight construction sites, overcast and shadow sites images, construction site with artifial light
- **Violation Types Represented:** Missing helmet, missing vest, fully compliant, missing gloves, missing footwear, missing goggles, missing harness
- **Annotation Tool:** Roboflow (web-based annotation)
- **Split**             Train(105) / Validation(30) / Test(15)

- **Class Distribution:**

  |----------------|-------|
  | Person         | [X]   |
  | Hardhat        | [X]   |
  | NO-Hardhat     | [X]   |
  | Safety Vest    | [X]   |
  | Gloves         | [X]   |
  | NO-Gloves      | [X]   |
  | Goggles        | [X]   |
  | NO-Goggles     | [X]   |
  | Harness        | [X]   |
  | NO-Harness     | [X]   |
  | NO-Safety Vest | [X]   |
  | Boots          | [X]   |
  | NO-Boot        | [X]   |

---

## Combined Dataset Summary

- **Total Images:** [2708 + 150]

---

**Note:**

- Custom images collected from publicly available web sources. So limited representation of nighttime or artificial lighting scenarios
- Limited visibilty of worn harness of the custom dataset.

---
