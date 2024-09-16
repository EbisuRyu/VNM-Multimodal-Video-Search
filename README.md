# Vietnamese Multimodal Video Search (Version 0.1)

**Vietnamese Multimodal Video Search** is an advanced platform designed to enable powerful video searches using multimodal data (visual, audio, and text) with a specific focus on Vietnamese content. Developed for the **HCM AI Challenge 2024**, this system allows users to perform sophisticated queries, detect events, and retrieve specific scenes, making it particularly useful for analyzing and understanding Vietnamese news videos.

### Key Features

| **Feature**                     | **Description**                                                                                  |
|----------------------------------|--------------------------------------------------------------------------------------------------|
| **Multimodal Data Support**      | Combines visual, audio, and textual inputs for comprehensive video search.                       |
| **Advanced Querying**            | Enables users to search for events, scenes, or keywords across multiple data types.              |
| **Event Detection**              | Automatically detects and indexes key events to simplify navigation within large datasets.        |
| **Scene Retrieval**              | Locates and retrieves specific video segments based on visual or textual cues.                   |
| **Vietnamese Content Focus**     | Optimized for the processing and analysis of Vietnamese-language videos.                         |

## Getting Started

<details>
<summary>Click to view the project directory structure</summary>

```
|- dataset 
   |- AIC_video
   |- beit
   |- blip
   |- clip
   |- color_palette
   |- distillation
   |- distilled_keyframe
   |- filter
   |- keyframe
   |- metadata
   |- scene_json
|- dict
   |- beit
   |- blip
   |- clip
   |- local
   |- metadata
   |- tag
|- extra 
   |- mmocr
   |- recognize-anything
   |- TransNetV2
   |- unilm
   |- yolo8x.pt
|- extraction
   |- beit
   |- blip
   |- clip
   |- distillation
   |- faiss
   |- filter
   |- metadata
   |- transnet
|- utils
   |- combine_module 
   |- embedding_based_search 
   |- filter
   |- object_color_search
   |- query_processing
   |- system_call
   |- temporal_search
   |- user_feedback
```

</details>

---

### Prerequisites

- Python 3.8
- All required dependencies are listed in the `requirements.txt` file.

### Installation

1. Create and activate a conda environment:

   ```bash
   conda create -n aic2024-env python=3.8
   conda activate aic2024-env
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

---

### Installing Faiss

For installing Faiss:

- On a GPU-enabled system:

   ```bash
   conda install -c pytorch -c nvidia faiss-gpu
   ```

- On a CPU-only system:

   ```bash
   conda install -c pytorch faiss-cpu
   ```

For further details, visit the [Faiss installation guide](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md).

---

### Installing CLIP

Ensure PyTorch 1.7.1+ is installed along with the required packages:

- For GPU-based systems:

   ```bash
   conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
   pip install ftfy regex tqdm open_clip_torch
   pip install git+https://github.com/openai/CLIP.git
   ```

- For CPU-only systems:

   ```bash
   conda install --yes -c pytorch pytorch=1.7.1 torchvision cpuonly
   pip install ftfy regex tqdm open_clip_torch
   pip install git+https://github.com/openai/CLIP.git
   ```

More information can be found in the [CLIP README](https://github.com/openai/CLIP/blob/main/README.md).

---

### Installing BLIP

To install BLIP, run:

```bash
pip install salesforce-lavis
```

Alternatively, you can build from source:

```bash
git clone https://github.com/salesforce/LAVIS.git
cd LAVIS
pip install -e .
```

For additional details, visit the [BLIP README](https://github.com/salesforce/LAVIS/blob/main/README.md).

---

### Installing BM25-Sparse

To install BM25:

```bash
pip install bm25s
```

For enhanced features like stemming and top-k selection:

```bash
pip install bm25s[full] PyStemmer jax[cpu]
```

Refer to the [BM25-Sparse README](https://github.com/xhluca/bm25s/blob/main/README.md) for more details.

---