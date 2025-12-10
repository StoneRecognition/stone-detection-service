# Module Comparison: Pros, Cons & Recommendations

This document analyzes all detection/segmentation modules for stone detection.

## Quick Comparison

| Module | Type | Speed | Detections | Best For |
|--------|------|-------|------------|----------|
| **GroundingDINO** | Detection | Fast | ~10-30 | Text-guided boxes |
| **RAM/RAM+** | Tagging | Fast | N/A | Scene understanding |
| **TAG2TEXT** | Caption+Tag | Fast | N/A | Context discovery |
| **SAM** | Segmentation | Slow | 100+ | Finding all objects |
| **SAM-HQ** | Segmentation | Slow | 100+ | High-quality edges |
| **Grounded-SAM** | Detect+Seg | Medium | 10-30 | Labeled masks |
| **SAM2** | Segmentation | Medium | 100+ | Video, temporal |
| **MobileSAM** | Segmentation | **Very Fast** | 80+ | Real-time apps |
| **YOLO-World** | Detection | **Very Fast** | 20-50 | Real-time detection |
| **OWL-ViT** | Detection | Medium | 10-30 | Small objects |
| **Florence-2** | Multi-task | Medium | 30+ | All-in-one |
| **Semantic-SAM** | Seg+Class | Slow | 50+ | Semantic labels |
| **SEEM** | Universal Seg | Slow | 50+ | Multi-modal |
| **OpenSeeD** | Open-vocab Seg | Slow | 50+ | Custom classes |
| **GenSAM** | Auto Seg | Slow | 50+ | Generalized |

---

## Detailed Analysis

### 1. GroundingDINO

**Purpose:** Text-guided object detection

**Pros:**
- Understands text prompts ("stone", "rock")
- Good for specific object classes
- Returns bounding boxes with confidence
- Fast inference

**Cons:**
- Misses small/subtle objects
- Limited to ~10-30 detections
- No segmentation masks
- Requires good prompt engineering

**Best Use:** First-pass detection, prompt-based filtering

---

### 2. RAM (Recognize Anything Model)

**Purpose:** Automatic image tagging

**Pros:**
- No prompts needed
- Discovers unexpected tags
- Fast inference
- Good scene understanding

**Cons:**
- No localization (just tags)
- May miss specific stone types
- Depends on training data

**Best Use:** Prompt discovery for other models

---

### 3. RAM+

**Purpose:** Enhanced tagging with more categories

**Pros:**
- More detailed tags than RAM
- Better fine-grained recognition
- Open-vocabulary

**Cons:**
- Larger model, slower
- Still no localization

**Best Use:** When RAM misses categories

---

### 4. TAG2TEXT

**Purpose:** Image captioning + tagging

**Pros:**
- Generates natural language captions
- Combined with tags
- Context understanding

**Cons:**
- Caption may not focus on stones
- Slower than RAM

**Best Use:** Understanding image context

---

### 5. SAM (Segment Anything Model)

**Purpose:** Automatic mask generation

**Pros:**
- Finds ALL objects (100+)
- High recall
- No prompts needed
- Good mask quality

**Cons:**
- No classification (just masks)
- Slow on large images
- Memory intensive

**Best Use:** Maximum object discovery

---

### 6. SAM-HQ

**Purpose:** Higher quality SAM masks

**Pros:**
- Better edge quality
- Improved small object handling
- Same API as SAM

**Cons:**
- Even slower than SAM
- Requires HQ checkpoint

**Best Use:** When mask quality matters

---

### 7. Grounded-SAM

**Purpose:** Combined detection + segmentation

**Pros:**
- Text-guided + masks
- Best of both worlds
- Labeled masks

**Cons:**
- Limited by GroundingDINO recall
- More complex pipeline

**Best Use:** Labeled segmentation masks

---

### 8. SAM2

**Purpose:** Next-generation SAM with video

**Pros:**
- Better architecture
- Video support
- Improved small objects

**Cons:**
- Requires SAM2 installation
- Different API

**Best Use:** Video processing, improved quality

---

### 9. MobileSAM

**Purpose:** Fast lightweight segmentation

**Pros:**
- **10x faster** than SAM
- Low memory usage
- Good for ensemble

**Cons:**
- Slightly lower quality
- Fewer masks

**Best Use:** Real-time, multiple passes

---

### 10. YOLO-World

**Purpose:** Real-time open-vocab detection

**Pros:**
- **Very fast** (real-time)
- Open vocabulary
- Easy to use

**Cons:**
- Detection only (no masks)
- May miss small objects

**Best Use:** Fast first-pass detection

---

### 11. OWL-ViT

**Purpose:** Open-vocabulary detection

**Pros:**
- Good for small objects
- Any text prompt
- Transformer-based

**Cons:**
- Slower than YOLO
- Requires HuggingFace

**Best Use:** Finding small stones

---

### 12. Florence-2

**Purpose:** Multi-task vision foundation

**Pros:**
- Detection + segmentation + caption
- Single model
- Microsoft quality

**Cons:**
- Large model
- Requires transformers

**Best Use:** All-in-one pipeline

---

### 13-16. Semantic-SAM, SEEM, OpenSeeD, GenSAM

**Purpose:** Advanced semantic segmentation

**Pros:**
- Semantic labels
- Open vocabulary
- Multi-granularity

**Cons:**
- Complex installation
- Require specific checkpoints
- Research-stage

**Best Use:** When semantic labels needed

---

## Recommended Combinations

### Option 1: Maximum Recall Pipeline
```
RAM -> SAM Auto -> Size Filter -> COCO Export
```
- RAM discovers tags
- SAM finds all objects (100+)
- Filter by size
- **Result:** ~40-50 stones

### Option 2: Fast + Accurate Pipeline
```
YOLO-World -> MobileSAM -> NMS -> Export
```
- YOLO-World: fast detection
- MobileSAM: fast masks
- **Result:** ~20-30 stones (fast)

### Option 3: Best Quality Pipeline
```
Florence-2 -> SAM-HQ -> Validation -> Export
```
- Florence-2: multi-task detection
- SAM-HQ: high-quality masks
- **Result:** ~30-40 stones (best quality)

### Option 4: Ensemble Pipeline (Recommended)
```
[GroundingDINO + YOLO-World + OWL-ViT] 
  -> Merge boxes 
  -> SAM-HQ masks 
  -> RAM validation
  -> COCO Export
```
- Multiple detectors for max recall
- SAM-HQ for quality
- RAM for verification
- **Result:** 50+ stones
