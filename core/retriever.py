import os
import torch
import numpy as np
import pandas as pd
import faiss
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoProcessor, AutoModel

# ============================================================
# Paths and Configurations
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "fashion_search_hnsw.faiss")
METADATA_PATH = os.path.join(BASE_DIR, "images_metadata.parquet")

# ============================================================
# 1. Load Data
# ============================================================
print("Loading FAISS index...")
if not os.path.exists(FAISS_INDEX_PATH):
    raise FileNotFoundError(f"FAISS index not found at {FAISS_INDEX_PATH}. Did you copy it from Colab?")
index = faiss.read_index(FAISS_INDEX_PATH)

print("Loading Metadata Database...")
if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}. Did you copy it from Colab?")
df_metadata = pd.read_parquet(METADATA_PATH)
df_metadata.set_index("faiss_id", inplace=True)

# ============================================================
# 2. Load Models
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Models onto {device.upper()} (This may take a minute)...")

# A. Normalization Model (Qwen2.5-0.5B-Instruct)
llm_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model = AutoModelForCausalLM.from_pretrained(
    llm_model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)
llm_model.eval()

# B. Embedding Model (BGE-M3)
embed_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=(device=="cuda"))

# C. Re-ranking Model (SigLIP)
siglip_model_name = "google/siglip-base-patch16-224"
siglip_processor = AutoProcessor.from_pretrained(siglip_model_name)
siglip_model = AutoModel.from_pretrained(siglip_model_name).to(device)
siglip_model.eval()

print("✅ All systems ready.\n")

# ============================================================
# 3. Pipeline Functions
# ============================================================

def normalize_text(raw_query):
    """Normalizes natural language into strict retrieval phrases."""
    system_prompt = """
You are a semantic query canonicalization model for a fashion search engine.
Convert natural language user queries into clean, pipeline-separated keywords for vector retrieval.

Rules:
1. Extract all clothing items, colors, accessories, and footwear.
2. Keep colors and attributes attached to their garments (e.g., "red tie", "white dress shirt").
3. PRESERVE styles, occasions, and vibes (e.g., "casual", "formal", "professional business attire", "weekend outfit").
4. PRESERVE locations and environments (e.g., "office", "city walk", "park bench").
5. Remove conversational filler (e.g., "Show me", "I want", "Someone wearing", "A person in").
6. Remove actions that are irrelevant to fashion (e.g., "sitting", "walking").
7. Output ONLY the canonical keywords separated by " | ".
"""
    
    base_examples = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Input: A person in a bright yellow raincoat.\nOutput:"},
        {"role": "assistant", "content": "bright yellow raincoat"},
        {"role": "user", "content": "Input: Professional business attire inside a modern office.\nOutput:"},
        {"role": "assistant", "content": "professional business attire | modern office"},
        {"role": "user", "content": "Input: Someone wearing a blue shirt sitting on a park bench.\nOutput:"},
        {"role": "assistant", "content": "blue shirt | park bench"},
        {"role": "user", "content": "Input: Casual weekend outfit for a city walk.\nOutput:"},
        {"role": "assistant", "content": "casual weekend outfit | city walk"},
        {"role": "user", "content": "Input: A red tie and a white shirt in a formal setting.\nOutput:"},
        {"role": "assistant", "content": "red tie | white shirt | formal setting"},
        {"role": "user", "content": f"Input: {raw_query}\nOutput:"}
    ]
    
    text = llm_tokenizer.apply_chat_template(base_examples, tokenize=False, add_generation_prompt=True)
    inputs = llm_tokenizer([text], return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = llm_model.generate(**inputs, max_new_tokens=64, do_sample=False)
    
    decoded = llm_tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
    cleaned = decoded.strip('"').strip("'")
    if cleaned.lower().startswith("output:"):
        cleaned = cleaned[7:].strip()
    return cleaned

def search_fashion(query, top_k=5, candidate_pool_size=100):
    print(f"\n🔍 Processing Query: '{query}'")
    
    # Step 1: Normalize
    canonical_query = normalize_text(query)
    print(f"🔹 Canonical: '{canonical_query}'")
    
    # Step 2: Embed
    embed_output = embed_model.encode([canonical_query], max_length=8192)
    query_vector = np.array([embed_output['dense_vecs'][0]], dtype=np.float32)
    
    # Step 3: FAISS Search
    distances, indices = index.search(query_vector, candidate_pool_size)
    candidate_faiss_ids = indices[0]
    
    # Filter out invalid IDs (e.g., -1 if FAISS doesn't have enough elements)
    valid_ids = [fid for fid in candidate_faiss_ids if fid != -1 and fid in df_metadata.index]
    
    if not valid_ids:
        print("❌ No matches found.")
        return []

    # Fetch Metadata from Pandas using index lookup (.loc preserves list order)
    candidates_df = df_metadata.loc[valid_ids].copy()
    candidates_df['faiss_id'] = candidates_df.index
    
    # Step 4: SigLIP Re-ranking
    print(f"🚀 Re-ranking {len(candidates_df)} candidates using SigLIP2...")
    candidate_images = []
    valid_records = []
    
    for _, row in candidates_df.iterrows():
        # Ensure path is relative to current directory if moving between OS/machines
        img_path = row['image_path']
        if not os.path.exists(img_path):
            base_name = os.path.basename(img_path)
            img_path = os.path.join(BASE_DIR, "..", "data", "val_test2020", "test", base_name)
            
        try:
            img = Image.open(img_path).convert("RGB")
            candidate_images.append(img)
            valid_records.append({"faiss_id": row['faiss_id'], "image_path": img_path, "caption": row['original_caption']})
        except Exception:
            continue
            
    if not candidate_images:
        return []
        
    inputs = siglip_processor(
        text=[query] * len(candidate_images),
        images=candidate_images,
        padding=True,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = siglip_model(**inputs)
        logits = outputs.logits_per_image.squeeze(-1)
        scores = torch.sigmoid(logits).cpu().numpy()
        
    for i, rec in enumerate(valid_records):
        rec['rerank_score'] = float(scores[i])
        
    ranked_results = sorted(valid_records, key=lambda x: x['rerank_score'], reverse=True)
    return ranked_results[:top_k]

# ============================================================
# 4. Interactive CLI
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("Welcome to the Fashion Retrieval Engine!")
    print("Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")
    
    while True:
        try:
            user_input = input("Enter fashion query: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                break
            if not user_input:
                continue
                
            results = search_fashion(user_input, top_k=5)
            
            print("\n" + "-"*40)
            print("🏆 Top Results:")
            for i, res in enumerate(results):
                print(f"[{i+1}] Score: {res['rerank_score']:.4f} | Path: {res['image_path']}")
                print(f"    Caption: {res['caption']}")
            print("-"*40 + "\n")
            
        except KeyboardInterrupt:
            break
