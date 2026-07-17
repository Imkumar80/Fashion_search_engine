# Intelligent Fashion Search Engine

A multimodal fashion image retrieval system built to understand complex, multi-attribute, and context-aware natural language queries. 

Unlike standard zero-shot vision-language models (like vanilla CLIP) which often struggle with compositional attributes (e.g., distinguishing a "red shirt with blue pants" from a "blue shirt with red pants"), this system utilizes a highly precise two-stage pipeline: a **semantic canonicalization + dense vector search** followed by a **CLIP-based visual re-ranking**.

## Architecture

The system is split into two distinct workflows: the offline indexing pipeline and the online retrieval pipeline.

### 1.Indexing Pipeline
*Runs once to process raw images into searchable representations.*

```text
                         Fashion Image Dataset
                                 │
                                 ▼
                  ┌────────────────────────────────┐
                  │ ① Qwen3-VL-2B-Instruct         │
                  │ Vision-Language Model          │
                  │ Image → Rich Caption           │
                  └────────────────────────────────┘
                                 │
                                 ▼
      "blue patterned jacket, white top, blue plaid trousers,
              brown beret on orange background"
                                 │
                                 ▼
                  ┌────────────────────────────────┐
                  │ ② Qwen2.5-0.5B-Instruct        │
                  │ Text Normalization             │
                  └────────────────────────────────┘
                                 │
                                 ▼
      blue patterned jacket | white top | blue plaid trousers | brown beret
                                 │
                                 ▼
                  ┌────────────────────────────────┐
                  │ ③ BGE-M3                       │
                  │ Text Encoder                   │
                  │ 1024-D Dense Vector            │
                  └────────────────────────────────┘
                                 │
               ┌─────────────────┴──────────────────┐
               │                                    │
               ▼                                    ▼
      ┌─────────────────────┐          ┌────────────────────────────┐
      │ FAISS HNSW Index    │          │ DataFrame / Parquet        │
      │ Dense Vectors       │          │ image_path                 │
      │ Vector Database     │          │ original_caption           │
      │ [0.12, 0.51, ...]   │          │ faiss_id                   │
      └─────────────────────┘          └────────────────────────────┘
```

### 2. Retrieval Pipeline
*Real-time search handling user queries.*

```text
                    User Query
                         │
                         ▼
     "blue jacket with white top"
                         │
                         ▼
           ┌──────────────────────────────┐
           │ ④ Qwen2.5-0.5B-Instruct      │
           │ Query Normalization          │
           └──────────────────────────────┘
                         │
                         ▼
               blue jacket | white top
                         │
                         ▼
           ┌──────────────────────────────┐
           │ ⑤ BGE-M3                     │
           │ Text Encoder                 │
           │ Query Embedding (1024-D)     │
           └──────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ FAISS HNSW           │
              │ Cosine Similarity    │
              │ Top-N Vector Search  │
              └──────────────────────┘
                         │
                         ▼
        Top-N Candidates (Metadata + Image Paths)
                         │
                         ▼
=========================================================
                    CLIP RE-RANKING
=========================================================
                         │
                         ▼
        ┌──────────────────────────────────┐
        │ SigLIP / CLIP Image & Text       │
        │ Cross-modal Similarity           │
        └──────────────────────────────────┘
                         │
                         ▼
                Final Ranked Images
```

---

## Project Structure

```text
Fashion_search_engine/
├── core/
│   ├── indexing_pipeline.ipynb   # Colab notebook for the offline indexing pipeline
│   ├── retriever.py              # CLI script for the online retrieval pipeline
│   ├── fashion_search_hnsw.faiss # Generated FAISS vector index
│   └── images_metadata.parquet   # Metadata mapping FAISS IDs to image paths/captions
├── data/
│   └── val_test2020/
│       └── test/                 # Raw fashion images go here
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Environment Setup

Create a virtual environment and install the required dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 2. Data Preparation

Because the FAISS index and Parquet metadata files were generated externally (e.g., in Google Colab), you need to place the raw image files in the specific directory expected by the local retriever.

Place all your `.jpg` or `.png` images into the following relative path:
`Fashion_search_engine/data/val_test2020/test/`

*Note: The `retriever.py` script is designed to automatically extract the image filenames from the Colab-generated absolute paths and remap them to your local `data/` directory.*

Ensure your pre-built index files are present in the `core/` directory:
- `core/fashion_search_hnsw.faiss`
- `core/images_metadata.parquet`

---

## Running the Search Engine

Run the interactive CLI retriever script:

```bash
cd core/
python retriever.py
```

1. The script will first load the FAISS index and Pandas DataFrame.
2. It will download/load the necessary Hugging Face models into GPU memory (Qwen2.5, BGE-M3, and SigLIP).
3. Once initialized, you will be prompted to enter a natural language fashion query.

**Example Queries to Try:**
- *"A person in a bright yellow raincoat."*
- *"Professional business attire inside a modern office."*
- *"Someone wearing a blue shirt sitting on a park bench."*
- *"Casual weekend outfit for a city walk."*
- *"A red tie and a white shirt in a formal setting."*

---

## Technical Approach & Trade-offs

### Why a Two-Stage Pipeline?
Applying a model like CLIP zero-shot across an entire dataset of millions of images is fast but often lacks semantic precision. CLIP behaves like a "bag of words" and struggles with compositional logic (e.g., which color applies to which garment, or separating background context from clothing). 

By structuring the pipeline into two stages:
1. **Stage 1 (High Recall):** We use **Qwen3-VL** to extract rich, structured attributes (garments, colors, styles, environment) and index them using **BGE-M3** (a state-of-the-art dense text retrieval model). This ensures we retrieve images that actually contain the requested items and vibe, without the vision-language confusion.
2. **Stage 2 (High Precision):** We use a **SigLIP/CLIP** model as a re-ranker over only the top 100 candidates. This step ensures visual relevance and aesthetic matching, acting purely on a highly refined subset.

### Scalability
- **Storage & Search:** The dense text embeddings are stored in a FAISS HNSW index, which scales to millions of vectors with sub-millisecond retrieval times.
- **Compute:** The heavy VLM extraction (Qwen3-VL) is strictly an *offline* process. The online pipeline only requires a lightweight LLM (Qwen2.5 0.5B), a text embedder (BGE-M3), and a small vision model (SigLIP base) for re-ranking, making real-time search extremely fast and computationally inexpensive (running comfortably on a 4GB VRAM GPU).

### Future Work
- **Expanding Context:** The `normalize_text` LLM prompt can be easily extended to parse and index metadata like weather conditions, cities, or seasonal vibes.
- **Improved Precision:** Incorporating object detection (e.g., bounding boxes for specific garments) during the indexing phase would allow for region-specific multi-vector search, vastly improving color-to-garment mapping precision.
