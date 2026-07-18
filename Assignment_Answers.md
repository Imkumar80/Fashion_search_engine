# Assignment Write-up: Multimodal Fashion & Context Retrieval

## 1. What were my early approaches? (All approaches we thought and tried)

When first tackling the problem of multimodal fashion retrieval, I considered and experimented with a few baseline approaches:

*   **Approach 1: Vanilla Zero-Shot Vision-Language Models (e.g., CLIP)**
    *   *Concept:* The most straightforward method was to use OpenAI's CLIP model. Pass all dataset images through the visual encoder, pass the user's natural language search query through the text encoder, and compute the cosine similarity to find the closest match.
    *   *Why I tried it:* It is the industry-standard baseline for zero-shot image retrieval, fast to implement, and requires no fine-tuning.
    *   *Problems Faced:* CLIP acts largely like a "bag of words" and suffers heavily from **compositionality issues**. For example, when queried with "a red shirt with blue pants", CLIP often retrieves a "blue shirt with red pants." Because both semantic concepts (red, blue, shirt, pants) are present in the image, the model ranks it highly, failing to map the specific attribute to the correct garment. It also struggles with complex spatial reasoning ("sitting on a park bench").
*   **Approach 2: Object Detection + Tagging + Keyword Search**
    *   *Concept:* Use an object detection model trained on fashion items to extract bounding boxes of garments, then pass those crops to a color/attribute classifier. These tags are then searched using BM25 or ElasticSearch.
    *   *Why I tried it:* It solves the compositionality problem because tags are strictly bound to specific garments.
    *   *Problems Faced:* It lacks semantic flexibility. If a user queries for a "casual weekend outfit" (a vibe) or complex environmental context ("professional business attire inside a modern office"), keyword matching completely fails because these aren't explicit, hardcoded tags. It is rigid and limits the user's natural language capability.

---

## 2. What is my final approach? (How I got a better solution & What changes I made)

To solve the trade-off between the semantic flexibility of CLIP and the precise compositional logic of hard-tagging, I changed my strategy entirely. I designed a **Two-Stage Hybrid Pipeline (Semantic Canonicalization + Visual Re-ranking)**. 

### Stage 1: The Indexer (Semantic Canonicalization)
Instead of extracting visual embeddings directly, we use a powerful Vision-Language Model (VLM) to act as a bridge.
1.  **Image to Text:** We run the dataset through **Qwen3-VL**, prompting it to generate a rich, highly detailed caption of every image (covering garments, colors, fit, vibe, and environment).
2.  **Normalization:** We pass this raw caption through a lightweight LLM (**Qwen2.5-0.5B**) to normalize the text into a clean, canonical string of attributes (e.g., `blue patterned jacket | white top | blue plaid trousers | modern office`).
3.  **Vectorization:** We encode this normalized string using a state-of-the-art dense text retrieval model (**BGE-M3**) and store it in a **FAISS HNSW** vector database.

### Stage 2: The Retriever (Search & Re-rank)
1.  **Text Search (High Recall):** When the user enters a query, we normalize it using Qwen2.5 and embed it using BGE-M3. We query the FAISS index to get the Top-100 candidates based purely on text-to-text semantic similarity.
2.  **Visual Re-ranking (High Precision):** We take these Top-100 candidates and run them through **SigLIP** (an improved version of CLIP). SigLIP calculates the cross-modal similarity between the original user query and the *actual candidate images*, re-ordering them to ensure the best visual and aesthetic match is at the very top.

### The Final Prompts Used

In my final Two-Stage Hybrid Pipeline, I used two distinct prompts to translate images into highly structured, searchable text.

#### Prompt 1: VLM Image Captioning (Qwen3-VL-2B-Instruct)
This prompt was used to extract a raw, highly detailed caption from every image in a zero-shot manner.
```text
You are a professional fashion image caption generator for an intelligent fashion search engine.
Describe the image in ONE clear, short, and accurate sentence.
Include ONLY the following if clearly visible:
- Upper body clothing with type and color (e.g., black shirt)
- Lower body clothing with type and color (e.g., blue jeans)
- Visible accessories with color (e.g., red tie, black hat)
- Background or environment if relevant (e.g., office, indoor, city street, park)
- Posture or action if visible (e.g., standing, walking, sitting)
Rules:
- Focus only on visible and factual details
- Do NOT guess, infer, or add extra information
- Do NOT describe emotions, style, or intent.
- Mention ONLY visible facts.
- Never guess hidden clothing.
- Never infer brands unless clearly visible.
- Never use adjectives like stylish, elegant, beautiful or fashionable.
- If something is uncertain, omit it.
- Start directly with the description.
Return ONLY one sentence.
```

#### Prompt 2: Semantic Text Normalization (Qwen2.5-0.5B-Instruct)
The raw caption from the VLM was then passed to a smaller LLM using this prompt (along with few-shot examples) to canonicalize the text into strict attributes.
```text
You are a semantic text normalization model for an intelligent fashion retrieval system.
Extract ONLY essential fashion-related retrieval phrases.
Rules:
1. Extract clothing items only if explicitly mentioned.
2. Keep colors attached to their garments as ONE unit.
3. Keep garment attributes attached to garments.
4. Extract accessories only if explicitly visible.
5. Extract footwear only if explicitly visible.
6. Extract meaningful environments only if relevant. If a setting has a descriptive color/quality, output it as ONE merged phrase.
7. Ignore actions (walking, standing, sitting, posing, running).
8. Ignore body parts and hairstyles.
9. Ignore camera viewpoints.
10. Ignore background people.
11. Never infer style, occasion, weather or emotions.
12. Never invent garments, colors, accessories, or settings not present in the input.
13. Remove duplicate information.
14. If multiple colors belong to the same garment, keep them attached to that garment.
15. Preserve the exact descriptive words used in the input.
16. Output ONLY retrieval phrases separated by ' | '.
17. Never output complete sentences.
```

---

## 3. How did I reach there? & How did I solve the problems?

I realized that forcing a single model to do both *hard logic mapping* (which color goes to which shirt) and *visual aesthetic matching* was causing the system to fail on complex queries. I solved this by separating the concerns:

*   **Solving Compositionality:** By translating the image into highly structured text first (Stage 1), we enforce strict logic. The BGE-M3 text embedder is excellent at understanding sentence structure, meaning it won't confuse a "red shirt" with a "blue shirt."
*   **Solving Semantic Flexibility:** By using a large VLM (Qwen3-VL) to generate the text, we capture complex context, "vibes", and environments (e.g., "office setting", "casual wear") which traditional object tagging misses.
*   **Solving Visual Precision:** Text isn't enough to capture fashion aesthetics. A purely text-based search might return an outfit that technically matches the description but looks awful. By adding SigLIP at the very end to re-rank a small subset of 100 images, we get the visual precision of a vision-language model without its typical hallucination/compositionality pitfalls.

---

## 4. Possible ways to solve this problem, tradeoffs, what’s good and when?

*   **Vanilla CLIP/SigLIP (Single Stage):**
    *   *Trade-offs:* Extremely fast indexing and retrieval. No intermediate models needed. However, suffers from low precision on multi-attribute queries and "bag of words" failures.
    *   *When to use:* Building a quick MVP or searching a dataset where queries are simple, single-object focused (e.g., just "dogs" or "cars").
*   **VLM Captioning + Traditional Keyword Search (BM25/ElasticSearch):**
    *   *Trade-offs:* Fixes compositionality but fails on semantic nuances. "Crimson blouse" won't match a query for "red shirt" unless explicit synonym logic is hardcoded.
    *   *When to use:* Traditional e-commerce platforms where users search via exact categorical filters and dropdowns.
*   **Two-Stage Pipeline (Chosen Approach):**
    *   *Trade-offs:* Indexing the dataset is computationally heavy and slower because it requires running every single raw image through a large VLM (Qwen3-VL). 
    *   *When to use:* When retrieval accuracy, precise compositional logic, and completely open-ended natural language understanding are the top priorities (like in a modern AI-first fashion search engine).

---

## 5. Scalability: Would your retrieval logic work if the dataset grew to 1 million images?

**Yes, absolutely.** The retrieval logic is inherently designed for massive scale because of the architectural split:

*   **Storage & Search:** The dense text embeddings are stored in a **FAISS HNSW index**. HNSW (Hierarchical Navigable Small World) is an approximate nearest neighbor search algorithm that scales effortlessly to billions of vectors with sub-millisecond retrieval times. Searching 1 million vectors takes practically the same time as searching 1,000.
*   **Compute:** The computationally expensive part (Qwen3-VL extracting captions) is strictly an *offline process* that happens only once when an image is ingested. During real-time retrieval, the system only embeds a short text query (using a lightweight model) and does a FAISS lookup.
*   **Re-ranking:** The SigLIP vision model only ever processes the Top-100 candidates returned by FAISS. Even with 1 million images in the database, the heavy lifting at runtime is capped at exactly 100 images, allowing it to run in real-time on standard consumer hardware.

### Code-Level Mapping of Scalability

Here is how the system's architecture supports massive scale in the code:

**1. Storage & Search (FAISS HNSW)**
In `core/indexing_pipeline.ipynb`, the FAISS index is explicitly configured to use `IndexHNSWFlat`. HNSW builds a graph-based structure that allows for logarithmic search time ($O(\log N)$) rather than linear search time, meaning searching 1,000,000 items is barely slower than searching 1,000.
```python
        # 4. FAISS index (HNSW for approximate search — scales better than flat index)
        embedding_dim = 1024
        USE_HNSW = True
        if USE_HNSW:
            base_index = faiss.IndexHNSWFlat(embedding_dim, 32, faiss.METRIC_INNER_PRODUCT)
        else:
            base_index = faiss.IndexFlatIP(embedding_dim)
            
        index = faiss.IndexIDMap(base_index)
```
At runtime in `core/retrieval_pipeline.ipynb`, this translates to a single sub-millisecond call that instantly filters the entire database down to a manageable size:
```python
    # Instantly searches the entire database for the top candidates
    distances, indices = index.search(query_vector, candidate_pool_size)
```

**2. Compute Efficiency (Offline Heavy-Lifting vs. Online Lightweight Querying)**
The computationally expensive VLM (`Qwen3-VL`) and LLM processing runs purely offline inside the `indexing_pipeline.ipynb` batching loops. 
During real-time retrieval (`core/retrieval_pipeline.ipynb`), the system never runs a massive generative model. It only runs the lightweight embedding model (`BGE-M3`) on a short text query, which takes milliseconds:
```python
def search_fashion(query, top_k=5, candidate_pool_size=100):
    # Only lightweight text processing happens online
    canonical_query = normalize_text(query)
    embed_output = embed_model.encode([canonical_query], max_length=8192)
```

**3. Bounded Re-ranking (Capping the Vision Model)**
Passing raw images through a vision-language model like SigLIP is slow. If we ran SigLIP on the entire database at runtime, it would take hours. Instead, in `core/retrieval_pipeline.ipynb`, the parameter `candidate_pool_size` hard-caps the heavy lifting at exactly 100 images, making runtime complexity $O(1)$ relative to the database size.
```python
def search_fashion(query, top_k=5, candidate_pool_size=100):
    # ... FAISS returns maximum 100 items ...
    
    # SigLIP heavy lifting is strictly capped at len(candidates_df) <= 100
    print(f"🚀 Re-ranking {len(candidates_df)} candidates using SigLIP...")
    
    inputs = siglip_processor(
        text=[query],
        images=candidate_images,  # This array never exceeds 100 images
        padding="max_length",
        return_tensors="pt"
    ).to(device)
```

---

## 6. Zero-Shot Capability: How well does the system handle descriptions it hasn't seen explicitly in a training label?

**The system handles unseen descriptions exceptionally well, acting with true zero-shot capability.**

*   Because we do not use fixed, pre-defined classes or rigid training labels, the vocabulary is unbounded. 
*   **VLM World Knowledge:** Qwen3-VL has vast world knowledge and can describe novel, rare, or highly specific fashion items (e.g., "gorpcore aesthetic", "Y2K style low-rise jeans") even if they aren't standard fashion terminology.
*   **Semantic Dense Embeddings:** BGE-M3 maps meaning to vector space, not exact strings. If a user searches for a "maroon jumper", the embedding model knows semantically that this is exceptionally close to a "dark red sweater" in a generated caption, handling synonyms and unseen phrasing combinations flawlessly.
*   **Visual Generalization:** SigLIP natively handles zero-shot visual matching, ensuring that novel visual attributes requested by the user are still evaluated correctly in the final ranking step.

### Code-Level Mapping of Zero-Shot Capabilities

Here is how those zero-shot capabilities map directly to the code in the pipeline notebooks:

**1. Unbounded Vocabulary via VLM (Zero-Shot Text Generation)**
Instead of forcing the images into predefined classes, we use **Qwen3-VL** to dynamically generate completely unbounded text for every image (`core/indexing_pipeline.ipynb`).
```python
    prompt = """
You are a professional fashion image caption generator for an intelligent fashion search engine.
... [Prompt Continues]
"""
    # Zero-shot generation without explicit training labels!
    with torch.no_grad():
        generated_ids = vlm_model.generate(**inputs, max_new_tokens=128, do_sample=False)
```

**2. Semantic Dense Embeddings (Handling Unseen Descriptions)**
We convert the textual attributes into a dense semantic vector so the system can match meaning rather than exact strings. 

In `core/indexing_pipeline.ipynb`, we embed the canonical captions offline:
```python
        # Step 3 : BGE-M3 Embedding (Batch)
        embedding_dict = embed_model.encode(normalized_captions, max_length=8192)
        embeddings = np.asarray(embedding_dict["dense_vecs"], dtype=np.float32)
```
At runtime in `core/retrieval_pipeline (1).ipynb`, we embed the user's natural language query using the exact same semantic model. Because BGE-M3 understands semantic similarity, a query for "maroon jumper" naturally finds vectors close to it in multidimensional space (like a "dark red sweater").

**3. Visual Generalization via SigLIP (Zero-Shot Visual Matching)**
Even after narrowing down to the top 100 semantically matching text results, the system takes the raw user query and the *actual images* to run a final zero-shot visual similarity score in `core/retrieval_pipeline (1).ipynb`:
```python
    # Pass the RAW user text and the RAW images to SigLIP
    inputs = siglip_processor(text=[query], images=candidate_images, padding="max_length", return_tensors="pt").to(device)

    # SigLIP natively evaluates the zero-shot alignment between the novel text and the pixels
    with torch.no_grad():
        outputs = siglip_model(**inputs)
        logits = outputs.logits_per_image.squeeze(-1)
        scores = torch.sigmoid(logits).cpu().numpy()
```

### How the Prompts Guide the Zero-Shot Models

**Prompt 1 (The "Objective Eye")** prevents VLM hallucination. VLMs naturally want to be conversational. By explicitly stating `Do NOT guess, infer, or add extra information` and `Never use adjectives like stylish, elegant, beautiful`, the VLM is forced to act purely as a factual pixel-to-text translator. 

**Prompt 2 (The "Logic Enforcer" & Few-Shot Learning)** solves the compositionality problem. Inside `normalize_texts_batch`, it uses **Few-Shot Prompting**:
```python
    base_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Input: A person wearing a bright yellow raincoat and black pants, standing outdoors on a city street.\nOutput:"},
        {"role": "assistant", "content": "yellow raincoat | black pants | city street"},
        # ... more examples
    ]
```
By providing explicit input/output pairs, the LLM stops trying to "think" about the 17 rules and simply mimics the exact formatting logic demonstrated. It standardizes the text, drops conversational noise, and outputs canonical strings (e.g., `red shirt | blue jeans`) that the BGE-M3 model can vectorize cleanly.

---

## 7. Approaches for Future Work

### Adding Locations (Cities, Places) and Weather
To make the search engine aware of environmental contexts like cities and weather, the system can be extended via **Hybrid Search (Vector + Metadata)** and **Context Injection**:

1.  **VLM Prompt Expansion (Indexing Time):** Update the `Qwen3-VL` prompt to explicitly extract weather conditions (e.g., sunny, raining, snowing) and location types (e.g., urban street, beach, office) directly from the image pixels.
2.  **Metadata Extraction (Indexing Time):** If the images have EXIF data (GPS coordinates, timestamps), use a reverse-geocoding API to tag the image with specific cities (e.g., "Paris," "Tokyo") and use historical weather APIs to tag the exact weather at that time. Store these tags in PostgreSQL as structured JSON columns.
3.  **Query Expansion (Retrieval Time):** When a user searches for *"What to wear today in New York"*, an LLM agent intercepts the query, calls a real-time weather API for New York (e.g., "Raining, 15°C"), and rewrites the query for the vector engine: *"raincoat | waterproof boots | umbrella | urban street"*.
4.  **Pre-filtering:** Use the structured metadata in PostgreSQL to hard-filter (e.g., `WHERE city = 'New York'`) before passing the remaining candidates to FAISS and SigLIP, combining the precision of SQL with the semantic fuzziness of vectors.

### Improving Precision
While the current Two-Stage pipeline is highly accurate, precision can be further improved by moving from zero-shot inference to domain-specific fine-tuning:

1.  **Fine-Tuning the Embedder (BGE-M3):** BGE-M3 is a generalized text model. By fine-tuning it using a **Triplet Loss** dataset (Anchor: User Query, Positive: Correct Canonical Caption, Negative: Incorrect Canonical Caption), the model will learn fashion-specific vector spaces (e.g., learning that "crimson" and "burgundy" are close, but "v-neck" and "crew neck" are far apart).
2.  **Fine-Tuning SigLIP:** SigLIP can be fine-tuned using LoRA (Low-Rank Adaptation) on a dedicated fashion dataset. This will teach the vision encoder to focus heavily on fabric textures, stitching, and garment fit rather than generic object recognition.
3.  **Hard Negative Mining:** The biggest threat to precision is compositionality (e.g., confusing "red shirt and blue pants" with "blue shirt and red pants"). We can train the models specifically on these "hard negatives" to severely penalize the network when it swaps attributes.
4.  **Granular Metadata Routing:** Instead of relying entirely on dense vectors, we can prompt Qwen2.5 to output structured JSON instead of a canonical string (e.g., `{"upper": {"color": "red", "type": "shirt"}}`). We can then use an exact-match search for colors/types and only use Vector/SigLIP search for the "vibe" and aesthetic ranking.
