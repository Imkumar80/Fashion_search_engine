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
      │ FAISS HNSW Index    │          │ PostgreSQL Database        │
      │ Dense Vectors       │          │ (Relational Data Storage)  │
      │ Vector Database     │          └─────────────┬──────────────┘
      │ [0.12, 0.51, ...]   │                        │ (Exported)
      └─────────────────────┘                        ▼
                                       ┌────────────────────────────┐
                                       │ DataFrame / Parquet        │
                                       │ image_path                 │
                                       │ original_caption           │
                                       │ faiss_id                   │
                                       └────────────────────────────┘
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
│   ├── retrieval.py              # CLI script for the online retrieval pipeline
│   ├── fashion_search_hnsw.faiss # Generated FAISS vector index
│   └── images_metadata.parquet   # Metadata mapping FAISS IDs to image paths/captions
├── data/
│   └── val_test2020/
│       └── test/                 # Raw fashion images go here
├── requirements.txt
└── README.md
```

---

## Local Setup Instructions

If you prefer to run this project locally (e.g., on a machine with a dedicated GPU), follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Fashion_search_engine.git
cd Fashion_search_engine
```

### 2. Set Up a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

### 3. Install Dependencies
Install the required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```
*Note: Depending on your system and hardware, you might need to install PyTorch separately to ensure it is configured correctly for your GPU (CUDA) or CPU. Refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).*

### 4. Data Preparation
Ensure the generated index (`fashion_search_hnsw.faiss`), metadata (`images_metadata.parquet`), and the image dataset are placed correctly within the project directory as referenced in the notebooks/scripts.

### 5. Running Locally
You can run the Jupyter notebooks (`indexing_pipeline.ipynb` and `retrieval_pipeline.ipynb`) using Jupyter Lab or Jupyter Notebook:
```bash
pip install jupyterlab
jupyter lab
```
Alternatively, if you have extracted the retrieval logic into `core/retrieval.py`, you can run it via the terminal:
```bash
python core/retrieval.py
```

---

## Google Colab Setup Instructions

This project was developed, indexed, and retrieved entirely within **Google Colab**, with all data and databases hosted in **Google Drive**.


### 1. Compute Requirements (Colab)
- **Indexing Phase:** The dataset of 3,200 images was indexed using an **A100 GPU** via Colab Compute Units. Utilizing batching, the entire heavy-lifting process (Qwen3-VL captioning + BGE-M3 embedding) took approximately **30-40 minutes**.
- **Retrieval Phase:** The online search pipeline is lightweight and was successfully run on Colab's **T4 Free Tier**.

### 2. Google Drive Data Preparation
To run the notebooks, you must mount your Google Drive to your Colab session. You can download the pre-computed FAISS index, metadata, and dataset directly from **[this Google Drive folder](https://drive.google.com/drive/folders/1AJW-zH023vryszWsiLJfq1OHdrp54qJW?usp=drive_link)**. 

Ensure your directory structure in your own Google Drive looks exactly like this after downloading:

```text
/content/drive/MyDrive/Fashion_search_engine/
├── fashion_search_hnsw.faiss       # The pre-computed FAISS vector index
├── images_metadata.parquet         # Metadata and canonical captions
└── data/
    └── val_test2020/
        └── test/                   # The raw .jpg / .png image dataset
```

### 3. Notebook Execution
To execute the pipelines in Colab:
1. Open the retrieval notebook (e.g., `retrieval_pipeline.ipynb`) in Google Colab.
2. Run the first cell to mount your Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Run the setup cells to download the required dependencies (Transformers, FAISS, Accelerate, etc.).

---

## Running the Search Engine in Colab

Inside the retrieval notebook, execute the cells sequentially:

1. The notebook will first load the FAISS index and Pandas DataFrame directly from your mounted Google Drive (`/content/drive/MyDrive/Fashion_search_engine/`).
2. It will download and load the necessary Hugging Face models into the T4 GPU memory (Qwen2.5, BGE-M3, and SigLIP).
3. Once initialized, you can modify the `query` variable in the search cell to enter a natural language fashion query.

**Example Queries to Try:**
- *"A person in a bright yellow raincoat."*
- *"Professional business attire inside a modern office."*
- *"Someone wearing a blue shirt sitting on a park bench."*
- *"Casual weekend outfit for a city walk."*
- *"A red tie and a white shirt in a formal setting."*

## Models Used

Here is a breakdown of the models powering the system, why they were chosen, and where to find them:

| Model | Pipeline Stage | Purpose | Link |
| --- | --- | --- | --- |
| **Qwen3-VL-2B-Instruct** | Indexing | Acts as a Vision-Language Model (VLM) to extract rich, descriptive, and factual captions from raw fashion images (garments, colors, styles, environment). | [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) |
| **Qwen2.5-0.5B-Instruct** | Indexing & Retrieval | Performs semantic text normalization. In indexing, it converts raw captions into structured keywords. In retrieval, it canonicalizes natural language user queries to match the indexed format. | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |
| **BGE-M3** | Indexing & Retrieval | State-of-the-art dense text embedding model used to convert the canonicalized text into 1024-D vectors for efficient and scalable FAISS similarity search. | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) |
| **SigLIP2-Base (Patch16-224)**| Retrieval | Used for cross-modal visual re-ranking (Stage 2). Evaluates the actual image-to-text similarity for the top 100 candidates retrieved by FAISS to ensure high precision and visual relevance. | [google/siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224) |

---

## Technical Approach & Trade-offs

### Why a Two-Stage Pipeline vs. Pure CLIP?
Standard zero-shot CLIP models are incredibly fast for image retrieval, but they have a major limitation: they act like a "bag of words." If you query "a blue shirt and red pants," CLIP often returns images of "a red shirt and blue pants" because it struggles with **compositional logic** (binding specific colors/attributes to specific garments). It also struggles to separate background context from the clothing itself.

To solve this, we structured the pipeline into two stages:

1. **Stage 1 (High Recall & Semantic Precision):** Instead of using vision-language embeddings for the initial search, we use **Qwen3-VL** to extract factual text captions and embed those using **BGE-M3** (a dense text model). This ensures the initial Top-100 results strictly contain the correct items in the correct colors.
2. **Stage 2 (High Precision & Visual Grounding):** Text search alone cannot evaluate aesthetic quality, fit, or visual layout. We apply a **SigLIP/CLIP** model to cross-examine the text query against the actual image pixels for only the top 100 candidates. This acts as a highly refined visual filter.

### The Role of SigLIP2
For the visual re-ranking step, we use **SigLIP2** (`google/siglip2-base-patch16-224`) rather than standard OpenAI CLIP. 
- **Standard CLIP** uses a softmax loss function, which requires pairwise comparisons across the entire batch, sometimes forcing the model to make artificial distinctions.
- **SigLIP** (Sigmoid Loss for Language Image Pre-Training) evaluates the image-text match independently using a sigmoid loss. This makes it far better at handling complex, multi-attribute descriptions, noisy data, and finer-grained details, making it the perfect re-ranker for complex fashion queries.

### Scalability
- **Storage & Search:** The dense text embeddings are stored in a FAISS HNSW index, which scales to millions of vectors with sub-millisecond retrieval times.
- **Compute:** The heavy VLM extraction (Qwen3-VL) is strictly an *offline* process. The online pipeline only requires a lightweight LLM (Qwen2.5 0.5B), a text embedder (BGE-M3), and a small vision model (SigLIP base) for re-ranking, making real-time search extremely fast and computationally inexpensive (running comfortably on a 4GB VRAM GPU).

### Future Work
- **Expanding Context:** The `normalize_text` LLM prompt can be easily extended to parse and index metadata like weather conditions, cities, or seasonal vibes.
- **Improved Precision:** Incorporating object detection (e.g., bounding boxes for specific garments) during the indexing phase would allow for region-specific multi-vector search, vastly improving color-to-garment mapping precision.
