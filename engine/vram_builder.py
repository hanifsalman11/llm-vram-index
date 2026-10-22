import os
import json
import re
import time
import urllib.request
import urllib.parse
from datetime import datetime, timezone

# ==========================================
# ENVIRONMENT CONFIGURATION
# ==========================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
SITE_BASE_URL = os.environ.get("SITE_BASE_URL", "https://hanifsalman11.github.io/llm-vram-index").rstrip("/")
INDEXNOW_KEY = os.environ.get("INDEXNOW_KEY", "b7a6c5d4e3f2a1b0c9d8e7f6a5b4c3d2").strip()
CUSTOM_MODEL = os.environ.get("CUSTOM_MODEL", "").strip()
PAGES_PER_RUN = int(os.environ.get("PAGES_PER_RUN", "3"))

DOCS_DIR = "docs"
DATA_DIR = "data"
PUBLISHED_FILE = os.path.join(DATA_DIR, "published_models.json")
MONETIZATION_FILE = os.path.join(DATA_DIR, "monetization.json")

# Format: (Model Name, Parameter Billions, Default Context Length, Architecture Family, Primary Use Case)
SEED_MODELS = [
    ("Llama 3.3 70B Instruct", 70.6, 128000, "Meta Llama-3 GQA", "Enterprise Reasoning & Coding"),
    ("DeepSeek R1 Distill Qwen 32B", 32.8, 128000, "Qwen2.5 Dense GQA", "Chain-of-Thought Math & Code"),
    ("Qwen 2.5 Coder 32B Instruct", 32.5, 128000, "Qwen2.5 Dense", "Local Software Engineering"),
    ("DeepSeek R1 Distill Llama 70B", 70.6, 128000, "Llama-3 GQA", "Deep Reasoning & Logic"),
    ("Mistral Large 2 123B Instruct", 123.0, 128000, "Mistral Dense GQA", "Multilingual & Agentic Workflows"),
    ("Qwen 2.5 72B Instruct", 72.7, 128000, "Qwen2.5 Dense", "Flagship Open-Weights Generalist"),
    ("Gemma 2 27B IT", 27.2, 8192, "Google Gemma-2 Interleaved", "High-Efficiency Consumer GPU"),
    ("Llama 3.1 8B Instruct", 8.0, 128000, "Meta Llama-3 GQA", "Fast Local Assistant & RAG"),
    ("Mistral Small 3 24B Instruct", 24.0, 32768, "Mistral Dense", "Low-Latency 16GB/24GB VRAM Tier"),
    ("Qwen 2.5 14B Instruct", 14.7, 128000, "Qwen2.5 Dense", "Mid-Range 12GB-16GB GPU Sweetspot"),
    ("Phi-4 14B Reasoning", 14.0, 16384, "Microsoft Dense", "STEM & Compact Local Inference"),
    ("Command R+ 104B (08-2024)", 104.0, 128000, "Cohere RAG Optimized", "Long-Context Enterprise RAG"),
    ("Mixtral 8x22B Instruct MoE", 141.0, 65536, "Sparse Mixture-of-Experts", "High-Throughput Multi-GPU MoE"),
    ("Mixtral 8x7B Instruct v0.1", 46.7, 32768, "Sparse Mixture-of-Experts", "Fast MoE Local Inference"),
    ("Llama 3.2 11B Vision Instruct", 10.7, 128000, "Multimodal Vision-Language", "Local Image & Document OCR"),
    ("Llama 3.2 90B Vision Instruct", 90.0, 128000, "Multimodal Vision-Language", "Enterprise Visual Reasoning"),
    ("Qwen 2 VL 72B Instruct", 72.0, 32768, "Multimodal Vision-Language", "High-Res Video & Document AI"),
    ("DeepSeek V2.5 236B MoE", 236.0, 128000, "DeepSeek MLA MoE", "Workstation Cluster Coding"),
    ("Codestral 22B v0.1", 22.2, 32768, "Mistral Code Specialist", "IDE Autocomplete & Refactoring"),
    ("StarCoder2 15B Instruct", 15.5, 16384, "BigCode Transformer", "Multi-Language Code Generation"),
    ("Yi 1.5 34B Chat", 34.0, 32768, "01.AI Dense GQA", "Creative Writing & Bilingual Chat"),
    ("Solar Pro 22B Preview", 22.0, 4096, "Upstage Depth-Upscaled", "Single 24GB GPU High-IQ Model"),
    ("Nemotron 4 70B Instruct", 70.6, 128000, "NVIDIA SteerLM Llama-3", "Reward-Tuned Alignment & Chat"),
    ("Aya Expanse 32B", 32.0, 128000, "Cohere Multilingual", "23-Language Local Translation"),
    ("Dolphin 2.9.4 Llama 3.1 70B", 70.6, 128000, "Uncensored Llama-3 Fine-Tune", "Unrestricted Local Roleplay & Code"),
    ("Hermes 3 Llama 3.1 70B", 70.6, 128000, "Nous Research Agentic", "Function Calling & Tool Use"),
    ("WizardLM 2 8x22B MoE", 141.0, 65536, "Sparse MoE", "Complex Instruction Following"),
    ("InternVL 2.5 38B Multimodal", 38.0, 32768, "Vision-Language Hybrid", "Scientific Chart & OCR Analysis"),
    ("Marco-o1 7B Reasoning", 7.6, 32768, "MCTS Chain-of-Thought", "8GB Budget GPU Reasoning"),
    ("SmolLM2 1.7B Instruct", 1.7, 8192, "HuggingFace Edge LLM", "Ultra-Low VRAM Edge & Laptop CPU")
]


def load_json(filepath, default):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Could not parse {filepath}: {e}")
    return default


def save_json(filepath, data):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def slugify(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    return re.sub(r"[\s-]+", "-", text).strip("-")


def safe_float(val, default=32.0):
    try:
        cleaned = re.sub(r"[^0-9.]", "", str(val))
        return float(cleaned) if cleaned else float(default)
    except Exception:
        return float(default)


def build_amazon_url(search_keywords, mon_cfg):
    domain = mon_cfg.get("marketplace_domain", "www.amazon.com")
    tag = mon_cfg.get("amazon_associate_tag", "YOUR_AMAZON_TAG-20")
    encoded_q = urllib.parse.quote_plus(search_keywords)
    return f"https://{domain}/s?k={encoded_q}&tag={tag}"


def discover_google_llm_queries():
    """Mengambil model LLM yang sedang banyak dicari kebutuhan VRAM-nya di Google Autocomplete US."""
    discovered = []
    seeds = [
        "vram requirements for ",
        "can rtx 4090 run ",
        "how much vram for "
    ]
    for seed in seeds:
        try:
            url = f"http://suggestqueries.google.com/complete/search?client=firefox&hl=en&gl=us&q={urllib.parse.quote(seed)}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                for suggestion in data[1]:
                    cleaned = suggestion.lower()
                    for pref in seeds:
                        cleaned = cleaned.replace(pref, "")
                    cleaned = re.sub(r"\b(vram|requirements|calculator|gpu|local|ollama|gguf)\b", "", cleaned).strip()
                    if 4 <= len(cleaned) <= 40:
                        param_match = re.search(r"(\d+(?:\.\d+)?)\s*b\b", cleaned, flags=re.IGNORECASE)
                        params_b = float(param_match.group(1)) if param_match else 32.0
                        discovered.append((cleaned.title(), params_b, 32768, "Transformer GQA", "Local AI Inference"))
        except Exception as e:
            print(f"[WARN] Google Suggest notice: {e}")
    return discovered


def build_deterministic_vram_data(model_tuple):
    """Mesin Fisika Tensor & Kuantisasi Otomatis (100% bebas gagal) jika Gemini API limit atau kosong."""
    model_name, params_b, max_ctx, arch, use_case = model_tuple

    # Perhitungan fisika bobot GGUF + KV Cache 8K Context + 0.8GB CUDA Buffer
    vram_fp16 = round((params_b * 2.0) + 2.2, 1)
    vram_q8 = round((params_b * 1.08) + 1.8, 1)
    vram_q4 = round((params_b * 0.62) + 1.5, 1)

    if vram_q4 > 48:
        gpu_rec = "Dual NVIDIA GeForce RTX 3090 / RTX 4090 24GB (48GB+ Total VRAM) or Mac Studio M2/M4 Ultra 128GB"
        gpu_query = "NVIDIA GeForce RTX 4090 24GB GDDR6X Graphics Card"
        gpu_price = "$1,699 - $2,499"
        mac_rec = "Apple Mac Studio / Mac Mini Pro Unified Memory Workstation"
        mac_query = "Apple Mac Studio M2 Ultra Unified Memory Workstation"
        mac_price = "$1,999 - $3,999"
    elif vram_q4 > 24:
        gpu_rec = "NVIDIA GeForce RTX 5090 32GB or Dual RTX 3090 24GB NVLink Rig"
        gpu_query = "NVIDIA GeForce RTX 3090 4090 24GB Graphics Card"
        gpu_price = "$999 - $2,199"
        mac_rec = "Apple Mac Mini M4 Pro (64GB Unified Memory)"
        mac_query = "Apple Mac Mini M4 Pro 64GB Unified Memory"
        mac_price = "$1,599 - $2,199"
    elif vram_q4 > 12:
        gpu_rec = "NVIDIA GeForce RTX 4090 24GB or RTX 4080 Super 16GB GDDR6X"
        gpu_query = "NVIDIA GeForce RTX 4080 Super 16GB Graphics Card"
        gpu_price = "$799 - $1,699"
        mac_rec = "Apple Mac Mini M4 Pro 24GB / 48GB Unified Memory"
        mac_query = "Apple Mac Mini M4 Pro 24GB Unified Memory"
        mac_price = "$1,299 - $1,699"
    else:
        gpu_rec = "NVIDIA GeForce RTX 4070 Ti Super 16GB or RTX 4060 Ti 16GB VRAM"
        gpu_query = "NVIDIA GeForce RTX 4060 Ti 16GB GDDR6 Graphics Card"
        gpu_price = "$449 - $799"
        mac_rec = "Apple Mac Mini M4 16GB / 24GB Unified Memory"
        mac_query = "Apple Mac Mini M4 24GB Unified Memory"
        mac_price = "$599 - $999"

    return {
        "model_name": model_name,
        "seo_title": f"{model_name} VRAM Requirements, GGUF Quantization & GPU Calculator (2026)",
        "meta_description": f"Calculate exact GPU VRAM and Unified Memory needed to run {model_name} ({params_b}B) in FP16 ({vram_fp16}GB), Q8_0 ({vram_q8}GB), and Q4_K_M ({vram_q4}GB) with Ollama or llama.cpp.",
        "specs": {
            "params_billions": params_b,
            "max_context": max_ctx,
            "architecture": arch,
            "use_case": use_case,
            "vram_fp16_gb": vram_fp16,
            "vram_q8_gb": vram_q8,
            "vram_q4_km_gb": vram_q4
        },
        "engineering_verdict": f"Running {model_name} ({params_b}B parameters, {arch}) at full unquantized FP16 precision requires {vram_fp16} GB of VRAM. Using 4-bit GGUF quantization (Q4_K_M) compresses the memory footprint down to {vram_q4} GB (including an 8,192-token KV cache), making it ideal for {gpu_rec}.",
        "hardware_recommendations": [
            {
                "badge": "Recommended Discrete GPU",
                "title": gpu_rec,
                "price_range": gpu_price,
                "amazon_query": gpu_query,
                "tech_note": f"Provides high-bandwidth GDDR6X/GDDR7 memory to run {model_name} at maximum tokens/sec without PCIe system RAM bottlenecks."
            },
            {
                "badge": "Best Silent Unified Memory Rig",
                "title": mac_rec,
                "price_range": mac_price,
                "amazon_query": mac_query,
                "tech_note": "Apple Silicon Unified Memory allocates up to 75% of system RAM directly to the Metal GPU for large GGUF/MLX models at under 65W wall power."
            },
            {
                "badge": "System RAM for CPU Layer Offloading",
                "title": "Corsair Vengeance / G.Skill 64GB (2x32GB) DDR5-6400 CL32 Memory Kit",
                "price_range": "$189 - $249",
                "amazon_query": "64GB 2x32GB DDR5 6400 CL32 Desktop Memory Kit",
                "tech_note": "Dual-channel DDR5-6400+ doubles token generation speed when partial layers overflow GPU VRAM into system memory."
            },
            {
                "badge": "Instant GGUF Model Loading Storage",
                "title": "2TB PCIe Gen4/Gen5 NVMe M.2 Internal SSD (7,400 MB/s)",
                "price_range": "$129 - $199",
                "amazon_query": "2TB PCIe Gen4 NVMe M.2 Internal Gaming SSD 7400MB",
                "tech_note": f"Loads {vram_q4}GB+ safetensors and GGUF model weights into VRAM in under 3 seconds via Microsoft DirectStorage / mmap."
            }
        ],
        "optimization_tips": [
            f"Use Q4_K_M or IQ4_XS quantization in Ollama / llama.cpp for {model_name}: it retains 99.1% of FP16 perplexity while cutting VRAM usage from {vram_fp16} GB down to {vram_q4} GB.",
            "Enable Flash Attention (-fa in llama.cpp or OLLAMA_FLASH_ATTENTION=1) and Q8_0 KV-Cache quantization to cut long-context memory overhead by 50%.",
            "Avoid splitting tensor layers across PCIe x4 risers when running multi-GPU setups; use `--split-mode row` with high-speed PCIe 4.0/5.0 x16 lanes.",
            "Set `--n-gpu-layers 99` to offload all transformer blocks to VRAM, and reduce layer count by 4–6 layers only if your context window exceeds 32,768 tokens."
        ],
        "faqs": [
            {
                "q": f"How much VRAM do I need to run {model_name} locally?",
                "a": f"At 4-bit quantization (Q4_K_M) with an 8K context window, {model_name} requires {vram_q4} GB of VRAM. For 8-bit quantization (Q8_0), you need {vram_q8} GB, and for full 16-bit (FP16) weights you need {vram_fp16} GB."
            },
            {
                "q": f"Can a single NVIDIA RTX 4090 (24GB) run {model_name}?",
                "a": (
                    f"Yes! Because {model_name} in Q4_K_M uses {vram_q4} GB of VRAM, all layers fit 100% inside a single 24GB RTX 3090 or RTX 4090 with room to spare for KV cache."
                    if vram_q4 <= 23.5
                    else f"At {vram_q4} GB for Q4_K_M, {model_name} exceeds a single 24GB GPU. You will either need aggressive IQ2_XS/IQ3_XS quantization, partial layer offloading to DDR5 system RAM, or a Dual-24GB GPU / 64GB Unified Memory setup."
                )
            },
            {
                "q": f"What is the exact Ollama command to run {model_name}?",
                "a": f"Install Ollama and execute `ollama run {slugify(model_name).split('-')[0]}` in your terminal. By default, Ollama pulls the 4-bit quantized GGUF build requiring approximately {vram_q4} GB of memory."
            }
        ]
    }


def generate_llm_data(model_tuple):
    """Memanggil Gemini 2.5 Flash API; jika gagal otomatis memakai Deterministic Tensor Engine."""
    if not GEMINI_API_KEY:
        return build_deterministic_vram_data(model_tuple)

    model_name, params_b, max_ctx, arch, use_case = model_tuple
    vram_fp16 = round((params_b * 2.0) + 2.2, 1)
    vram_q8 = round((params_b * 1.08) + 1.8, 1)
    vram_q4 = round((params_b * 0.62) + 1.5, 1)

    prompt = f"""
You are a Principal AI Systems & GPU Hardware Architect. Create an accurate 2026 VRAM sizing, GGUF quantization, and hardware build guide for running the LLM: "{model_name}" ({params_b}B parameters).
Baseline calculations: FP16={vram_fp16}GB, Q8_0={vram_q8}GB, Q4_K_M={vram_q4}GB, Max Context={max_ctx}, Architecture={arch}.
Return ONLY a valid JSON object matching this exact schema:
{{
  "model_name": "{model_name}",
  "seo_title": "{model_name} VRAM Requirements, GGUF Quantization & GPU Calculator (2026)",
  "meta_description": "Calculate exact GPU VRAM and Unified Memory needed to run {model_name} ({params_b}B) in FP16, Q8_0, and Q4_K_M with Ollama or llama.cpp.",
  "specs": {{
    "params_billions": {params_b},
    "max_context": {max_ctx},
    "architecture": "{arch}",
    "use_case": "{use_case}",
    "vram_fp16_gb": {vram_fp16},
    "vram_q8_gb": {vram_q8},
    "vram_q4_km_gb": {vram_q4}
  }},
  "engineering_verdict": "3-sentence technical summary explaining exact GB of VRAM needed for FP16 vs Q4_K_M and which NVIDIA RTX GPU or Apple Silicon Mac runs it best.",
  "hardware_recommendations": [
    {{
      "badge": "Recommended Discrete GPU",
      "title": "Exact NVIDIA RTX GPU model recommendation",
      "price_range": "$799 - $1,999",
      "amazon_query": "NVIDIA GeForce RTX 4090 24GB Graphics Card",
      "tech_note": "1-sentence memory bandwidth & tokens/sec explanation"
    }},
    {{
      "badge": "Best Unified Memory System",
      "title": "Exact Apple Mac Mini M4 Pro or Mac Studio recommendation",
      "price_range": "$1,299 - $2,999",
      "amazon_query": "Apple Mac Mini M4 Pro Unified Memory",
      "tech_note": "1-sentence Metal GPU unified memory explanation"
    }},
    {{
      "badge": "High-Speed System RAM (CPU Offload)",
      "title": "64GB (2x32GB) DDR5-6400 CL32 Desktop Memory Kit",
      "price_range": "$189 - $249",
      "amazon_query": "64GB DDR5 6400 CL32 Desktop Memory Kit",
      "tech_note": "1-sentence explanation on DDR5 bandwidth for layer offloading"
    }},
    {{
      "badge": "Fast Model Weight Storage",
      "title": "2TB PCIe Gen4/Gen5 NVMe M.2 SSD (7,400 MB/s)",
      "price_range": "$129 - $199",
      "amazon_query": "2TB PCIe Gen4 NVMe M.2 Internal SSD",
      "tech_note": "1-sentence explanation on fast safetensors/GGUF mmap loading"
    }}
  ],
  "optimization_tips": [
    "Tip 1 regarding GGUF quantization choice",
    "Tip 2 regarding Flash Attention and KV cache quantization",
    "Tip 3 regarding GPU layer offloading",
    "Tip 4 regarding power limits and cooling"
  ],
  "faqs": [
    {{"q": "How much VRAM do I need to run {model_name}?", "a": "Exact GB numbers for FP16, Q8_0, and Q4_K_M."}},
    {{"q": "Can an NVIDIA RTX 4090 (24GB) run {model_name}?", "a": "Direct technical answer with expected tokens per second."}},
    {{"q": "Is Mac Unified Memory good for running {model_name}?", "a": "Direct technical answer comparing Apple Silicon bandwidth."}}
  ]
}}
"""

    for model_id in ["gemini-2.5-flash", "gemini-2.0-flash"]:
        try:
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={GEMINI_API_KEY}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.2,
                    "responseMimeType": "application/json"
                }
            }
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=35) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                raw_text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
                raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
                raw_text = re.sub(r"\s*```$", "", raw_text)
                parsed = json.loads(raw_text)
                if "specs" in parsed and "hardware_recommendations" in parsed:
                    return parsed
        except Exception as e:
            print(f"[WARN] Gemini ({model_id}) notice for '{model_name}': {e}")

    return build_deterministic_vram_data(model_tuple)


def build_vram_html_page(data, slug, mon_cfg):
    model_name = data.get("model_name", "Local LLM")
    sp = data.get("specs", {})
    params_b = round(safe_float(sp.get("params_billions", 32.0), 32.0), 1)
    max_ctx = int(safe_float(sp.get("max_context", 32768), 32768))
    arch = sp.get("architecture", "Transformer GQA")
    use_case = sp.get("use_case", "Local Inference")

    header_cta = mon_cfg.get("header_cta", {})
    header_cta_url = build_amazon_url(header_cta.get("search_query", "NVIDIA+GeForce+RTX+4090+Graphics+Card"), mon_cfg)

    schema_graph = {
        "@context": "https://schema.org",
        "@graph": [
            {
                "@type": "SoftwareApplication",
                "name": f"{model_name} VRAM & GPU Sizing Calculator",
                "applicationCategory": "DeveloperApplication",
                "operatingSystem": "Windows, Linux, macOS",
                "offers": {"@type": "Offer", "price": "0", "priceCurrency": "USD"}
            },
            {
                "@type": "FAQPage",
                "mainEntity": [
                    {
                        "@type": "Question",
                        "name": f.get("q", ""),
                        "acceptedAnswer": {"@type": "Answer", "text": f.get("a", "")}
                    } for f in data.get("faqs", [])
                ]
            }
        ]
    }

    hw_html = ""
    for item in data.get("hardware_recommendations", []):
        amz_url = build_amazon_url(item.get("amazon_query", item.get("title", "RTX 4090 GPU")), mon_cfg)
        hw_html += f"""
        <div class="bg-neutral-900 border border-neutral-800 rounded-xl p-5 flex flex-col justify-between hover:border-lime-500/60 transition">
          <div>
            <div class="flex items-center justify-between gap-2 mb-2">
              <span class="text-xs font-bold uppercase tracking-wider bg-lime-500/10 text-lime-400 border border-lime-500/20 px-2.5 py-0.5 rounded">{item.get('badge', 'AI Hardware')}</span>
              <span class="text-sm font-bold text-emerald-400">{item.get('price_range', '$199+')}</span>
            </div>
            <h3 class="text-base font-bold text-white mt-2">{item.get('title', '')}</h3>
            <p class="text-xs text-neutral-400 mt-2 leading-relaxed">🖥️ {item.get('tech_note', '')}</p>
          </div>
          <a href="{amz_url}" target="_blank" rel="nofollow sponsored" class="mt-5 block text-center bg-lime-500 hover:bg-lime-400 text-neutral-950 font-extrabold py-2.5 px-4 rounded-lg text-sm transition shadow-lg">
            Check Price on Amazon →
          </a>
        </div>"""

    # Build quantization reference table
    quant_tiers = [
        ("FP16 / BF16 (Unquantized)", 2.00, "100% Baseline (Lossless)", "Multi-GPU Server / 80GB A100"),
        ("Q8_0 (8-Bit GGUF)", 1.08, "99.9% (Indistinguishable)", "High-VRAM Workstation / Mac Studio"),
        ("Q6_K (6-Bit K-Quant)", 0.82, "99.6% (Near-Lossless)", "Enthusiast 24GB–48GB VRAM"),
        ("Q5_K_M (5-Bit K-Quant)", 0.71, "99.3% (High Accuracy)", "Balanced Quality & Speed"),
        ("Q4_K_M (4-Bit Recommended)", 0.62, "98.8% (Sweet Spot)", "Consumer RTX GPUs & Mac Mini Pro"),
        ("IQ3_XS (3-Bit Importance Matrix)", 0.46, "96.2% (Noticeable Compression)", "Tight VRAM Budget Fit")
    ]
    quant_rows_html = ""
    for q_name, b_factor, quality, target_hw in quant_tiers:
        w_gb = round(params_b * b_factor, 1)
        tot_gb = round(w_gb + 1.5, 1)
        quant_rows_html += f"""
        <tr class="border-b border-neutral-800 hover:bg-neutral-800/40">
          <td class="py-3 px-4 font-bold text-white">{q_name}</td>
          <td class="py-3 px-4 text-neutral-300">{w_gb} GB</td>
          <td class="py-3 px-4 font-bold text-lime-400">{tot_gb} GB</td>
          <td class="py-3 px-4 text-emerald-400">{quality}</td>
          <td class="py-3 px-4 text-xs text-neutral-400">{target_hw}</td>
        </tr>"""

    tips_html = "".join([
        f"<li class='flex items-start gap-3 bg-neutral-900 border border-neutral-800 p-4 rounded-xl text-sm text-neutral-200'><span class='text-lime-400 font-bold'>⚡</span><span>{t}</span></li>"
        for t in data.get("optimization_tips", [])
    ])

    faqs_html = "".join([
        f"<div class='bg-neutral-900 border border-neutral-800 rounded-xl p-5'><h3 class='font-bold text-white text-base mb-2'>{f.get('q', '')}</h3><p class='text-sm text-neutral-300 leading-relaxed'>{f.get('a', '')}</p></div>"
        for f in data.get("faqs", [])
    ])

    canonical_url = f"{SITE_BASE_URL}/{slug}.html"
    amz_domain = mon_cfg.get("marketplace_domain", "www.amazon.com")
    amz_tag = mon_cfg.get("amazon_associate_tag", "YOUR_AMAZON_TAG-20")

    html = f"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{data.get('seo_title', model_name)}</title>
  <meta name="description" content="{data.get('meta_description', '')}">
  <link rel="canonical" href="{canonical_url}">
  <meta property="og:title" content="{data.get('seo_title', model_name)}">
  <meta property="og:description" content="{data.get('meta_description', '')}">
  <meta property="og:type" content="article">
  <meta property="og:url" content="{canonical_url}">
  <script src="https://cdn.tailwindcss.com"></script>
  <script type="application/ld+json">{json.dumps(schema_graph)}</script>
</head>
<body class="bg-neutral-950 text-neutral-100 antialiased min-h-screen flex flex-col">
  <header class="border-b border-neutral-800 bg-neutral-900/80 backdrop-blur sticky top-0 z-30">
    <div class="max-w-5xl mx-auto px-4 py-3.5 flex items-center justify-between">
      <a href="index.html" class="text-lg font-black tracking-tight text-white flex items-center gap-2">
        <span class="bg-lime-500 text-neutral-950 px-2 py-0.5 rounded font-black">VRAM</span>CalcIndex
      </a>
      <a href="{header_cta_url}" target="_blank" rel="nofollow sponsored" class="text-xs sm:text-sm bg-lime-500 hover:bg-lime-400 text-neutral-950 font-extrabold px-3.5 py-2 rounded-lg transition">
        {header_cta.get('cta_text', 'Shop Local AI GPUs →')}
      </a>
    </div>
  </header>

  <main class="max-w-5xl mx-auto px-4 py-10 flex-1 w-full">
    <div class="mb-8">
      <div class="flex flex-wrap items-center gap-2.5 mb-3">
        <span class="text-xs font-extrabold uppercase tracking-widest bg-lime-500 text-neutral-950 px-3 py-1 rounded-full">{params_b}B Parameters</span>
        <span class="text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-3 py-1 rounded-full">{arch}</span>
        <span class="text-xs font-semibold bg-neutral-800 text-neutral-300 px-3 py-1 rounded-full">Max Context: {max_ctx:,} Tokens</span>
      </div>
      <h1 class="text-3xl sm:text-4xl font-black text-white leading-tight">{data.get('seo_title', model_name)}</h1>
      <div class="mt-4 bg-neutral-900 border-l-4 border-lime-500 p-4 rounded-r-xl text-sm text-neutral-200 leading-relaxed">
        <strong>AI Hardware Benchmark Verdict:</strong> {data.get('engineering_verdict', '')}
      </div>
    </div>

    <!-- INTERACTIVE VRAM, QUANTIZATION & TOKENS/SEC CALCULATOR -->
    <section class="bg-neutral-900 border border-neutral-800 rounded-2xl p-6 sm:p-8 mb-12 shadow-2xl">
      <h2 class="text-xl font-extrabold text-white mb-2">Interactive {model_name} VRAM & Token Speed Calculator</h2>
      <p class="text-xs text-neutral-400 mb-6">Select your GGUF/EXL2 quantization precision, context window length, and hardware tier to calculate exact VRAM usage and estimated generation speed.</p>

      <div class="grid md:grid-cols-2 gap-8 items-center">
        <div class="space-y-5">
          <div>
            <label for="quantSelect" class="block text-sm font-medium text-neutral-300 mb-1.5">Quantization Precision (GGUF / AWQ / EXL2)</label>
            <select id="quantSelect" class="w-full bg-neutral-950 border border-neutral-700 rounded-lg px-3.5 py-2.5 text-sm text-white focus:outline-none focus:border-lime-500">
              <option value="2.00">FP16 / BF16 — 16-Bit Full Precision</option>
              <option value="1.08">Q8_0 — 8-Bit Extreme Quality</option>
              <option value="0.82">Q6_K — 6-Bit Near-Lossless</option>
              <option value="0.71">Q5_K_M — 5-Bit High Accuracy</option>
              <option value="0.62" selected>Q4_K_M — 4-Bit Recommended Sweet Spot</option>
              <option value="0.46">IQ3_XS — 3-Bit Low-VRAM Compression</option>
            </select>
          </div>

          <div>
            <div class="flex justify-between text-sm font-medium mb-1.5">
              <label for="ctxSlider" class="text-neutral-300">Context Window Size (KV Cache Tokens)</label>
              <span class="text-lime-400 font-bold"><span id="ctxVal">8,192</span> Tokens</span>
            </div>
            <input id="ctxSlider" type="range" min="2048" max="65536" step="2048" value="8192" class="w-full accent-lime-500 cursor-pointer">
          </div>

          <div>
            <label for="gpuTierSelect" class="block text-sm font-medium text-neutral-300 mb-1.5">Your Target GPU / Unified Memory Hardware</label>
            <select id="gpuTierSelect" class="w-full bg-neutral-950 border border-neutral-700 rounded-lg px-3.5 py-2.5 text-sm text-white focus:outline-none focus:border-lime-500">
              <option value="12|504">NVIDIA RTX 4070 Super (12GB VRAM | 504 GB/s)</option>
              <option value="16|736">NVIDIA RTX 4080 Super / 4070 Ti Super (16GB VRAM | 736 GB/s)</option>
              <option value="24|1008" selected>NVIDIA RTX 4090 / RTX 3090 (24GB VRAM | 1,008 GB/s)</option>
              <option value="32|1792">NVIDIA RTX 5090 (32GB GDDR7 VRAM | 1,792 GB/s)</option>
              <option value="48|1008">Dual NVIDIA RTX 3090 / 4090 Rig (48GB Total VRAM)</option>
              <option value="64|273">Apple Mac Mini M4 Pro (64GB Unified Memory | 273 GB/s)</option>
              <option value="128|800">Apple Mac Studio Ultra (128GB Unified Memory | 800 GB/s)</option>
            </select>
          </div>
        </div>

        <div class="grid grid-cols-2 gap-4">
          <div class="bg-neutral-950 border border-lime-500/30 rounded-xl p-4">
            <span class="text-xs uppercase text-lime-400 font-bold">Total VRAM Required</span>
            <div class="text-2xl sm:text-3xl font-black text-white mt-1"><span id="totalVramOut">0.0</span> <span class="text-sm font-normal text-neutral-400">GB</span></div>
            <p id="weightsBreakdownOut" class="text-xs text-neutral-500 mt-1">Weights + KV Cache + CUDA</p>
          </div>

          <div class="bg-neutral-950 border border-emerald-500/30 rounded-xl p-4">
            <span class="text-xs uppercase text-emerald-400 font-bold">Est. Generation Speed</span>
            <div class="text-2xl sm:text-3xl font-black text-emerald-400 mt-1">~<span id="tpsOut">0</span> <span class="text-sm font-normal text-neutral-400">tok/s</span></div>
            <p id="fitStatusOut" class="text-xs text-emerald-400 font-semibold mt-1">Fits 100% in VRAM</p>
          </div>
        </div>
      </div>

      <div class="mt-6 bg-lime-950/40 border border-lime-500/30 rounded-xl p-4 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div id="hardwareAdviceBanner" class="text-sm text-lime-100 font-medium"></div>
        <a id="dynamicGpuBtn" href="{header_cta_url}" target="_blank" rel="nofollow sponsored" class="shrink-0 bg-lime-500 hover:bg-lime-400 text-neutral-950 font-extrabold px-4 py-2.5 rounded-lg text-xs uppercase tracking-wider transition">
          Shop Recommended GPU on Amazon →
        </a>
      </div>
    </section>

    <!-- 4-CARD HARDWARE RECOMMENDATIONS -->
    <section class="mb-12">
      <div class="flex items-center justify-between mb-4">
        <h2 class="text-2xl font-extrabold text-white">Recommended Hardware Builds for {model_name}</h2>
        <span class="text-xs text-neutral-400">High-Bandwidth Local Inference Components</span>
      </div>
      <div class="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">{hw_html}</div>
    </section>

    <!-- QUANTIZATION COMPARISON TABLE -->
    <section class="mb-12">
      <h2 class="text-2xl font-extrabold text-white mb-4">{model_name} Quantization & Memory Footprint Matrix</h2>
      <div class="overflow-x-auto bg-neutral-900 border border-neutral-800 rounded-xl">
        <table class="w-full text-left border-collapse text-sm">
          <thead>
            <tr class="border-b border-neutral-800 bg-neutral-950/60 text-neutral-400 uppercase text-xs">
              <th class="py-3.5 px-4">Quantization Format</th>
              <th class="py-3.5 px-4">Model Weights Size</th>
              <th class="py-3.5 px-4 text-lime-400">Total VRAM (w/ 8K Context)</th>
              <th class="py-3.5 px-4 text-emerald-400">Perplexity Retention</th>
              <th class="py-3.5 px-4">Target Hardware Tier</th>
            </tr>
          </thead>
          <tbody>{quant_rows_html}</tbody>
        </table>
      </div>
    </section>

    <!-- OPTIMIZATION TIPS -->
    <section class="mb-12">
      <h2 class="text-2xl font-extrabold text-white mb-4">Ollama & llama.cpp Performance Tuning Tips</h2>
      <ul class="space-y-3">{tips_html}</ul>
    </section>

    <!-- FAQ -->
    <section class="mb-12">
      <h2 class="text-2xl font-extrabold text-white mb-5">Frequently Asked Questions ({model_name})</h2>
      <div class="space-y-4">{faqs_html}</div>
    </section>
  </main>

  <footer class="border-t border-neutral-800 py-8 text-center text-xs text-neutral-500 px-4">
    <p>© {datetime.now(timezone.utc).year} VRAMCalcIndex. Independent Local LLM & GPU Hardware Benchmark Database. As an Amazon Associate, we earn from qualifying purchases.</p>
  </footer>

  <script>
    const paramsB = {params_b};
    const amzDomain = "{amz_domain}";
    const amzTag = "{amz_tag}";

    const quantSelect = document.getElementById('quantSelect');
    const ctxSlider = document.getElementById('ctxSlider');
    const gpuTierSelect = document.getElementById('gpuTierSelect');
    const ctxVal = document.getElementById('ctxVal');
    const totalVramOut = document.getElementById('totalVramOut');
    const weightsBreakdownOut = document.getElementById('weightsBreakdownOut');
    const tpsOut = document.getElementById('tpsOut');
    const fitStatusOut = document.getElementById('fitStatusOut');
    const hardwareAdviceBanner = document.getElementById('hardwareAdviceBanner');
    const dynamicGpuBtn = document.getElementById('dynamicGpuBtn');

    function recalculateVram() {{
      const bFactor = parseFloat(quantSelect.value);
      const ctxTokens = parseInt(ctxSlider.value, 10);
      const gpuParts = gpuTierSelect.value.split('|');
      const availVram = parseFloat(gpuParts[0]);
      const memBandwidth = parseFloat(gpuParts[1]);

      ctxVal.textContent = ctxTokens.toLocaleString();

      const weightsGb = paramsB * bFactor;
      const kvCacheGb = (ctxTokens / 8192.0) * (paramsB * 0.022);
      const cudaOverheadGb = 0.8;
      const totalRequiredGb = weightsGb + kvCacheGb + cudaOverheadGb;

      totalVramOut.textContent = totalRequiredGb.toFixed(1);
      weightsBreakdownOut.textContent = `${{weightsGb.toFixed(1)}}GB Weights + ${{kvCacheGb.toFixed(1)}}GB KV`;

      if (totalRequiredGb <= availVram) {{
        const estTps = Math.min(145, Math.max(4, Math.round((memBandwidth * 0.68) / Math.max(1.5, weightsGb))));
        tpsOut.textContent = estTps;
        fitStatusOut.textContent = `✅ Fits 100% in ${{availVram}}GB VRAM`;
        fitStatusOut.className = "text-xs text-emerald-400 font-semibold mt-1";
      }} else {{
        const offloadRatio = availVram / totalRequiredGb;
        const estTps = Math.max(1, Math.round(12 * offloadRatio));
        tpsOut.textContent = estTps;
        fitStatusOut.textContent = `⚠️ Needs +${{(totalRequiredGb - availVram).toFixed(1)}}GB CPU RAM Offload`;
        fitStatusOut.className = "text-xs text-amber-400 font-semibold mt-1";
      }}

      let searchHw = "NVIDIA+GeForce+RTX+4090+24GB+Graphics+Card";
      if (totalRequiredGb <= 15.5) {{
        searchHw = "NVIDIA+GeForce+RTX+4070+Ti+Super+16GB+Graphics+Card";
        hardwareAdviceBanner.innerHTML = `🚀 <strong>Optimal Build:</strong> Requires <strong>${{totalRequiredGb.toFixed(1)}} GB VRAM</strong>. Runs at full speed on a single <strong>16GB NVIDIA RTX 4070 Ti Super / RTX 4080 Super</strong>.`;
      }} else if (totalRequiredGb <= 23.8) {{
        searchHw = "NVIDIA+GeForce+RTX+4090+24GB+Graphics+Card";
        hardwareAdviceBanner.innerHTML = `🚀 <strong>Optimal Build:</strong> Requires <strong>${{totalRequiredGb.toFixed(1)}} GB VRAM</strong>. Fits completely inside a single <strong>24GB NVIDIA RTX 3090 / RTX 4090</strong>.`;
      }} else if (totalRequiredGb <= 31.8) {{
        searchHw = "NVIDIA+GeForce+RTX+5090+32GB+Graphics+Card";
        hardwareAdviceBanner.innerHTML = `🚀 <strong>Optimal Build:</strong> Requires <strong>${{totalRequiredGb.toFixed(1)}} GB VRAM</strong>. Ideal for an <strong>NVIDIA RTX 5090 32GB</strong> or <strong>Mac Mini M4 Pro 64GB</strong>.`;
      }} else {{
        searchHw = "Apple+Mac+Studio+Unified+Memory+Workstation";
        hardwareAdviceBanner.innerHTML = `🚀 <strong>Workstation Tier:</strong> Requires <strong>${{totalRequiredGb.toFixed(1)}} GB Memory</strong>. Best paired with a <strong>Dual RTX 24GB GPU Rig (48GB)</strong> or <strong>Mac Studio Unified Memory</strong>.`;
      }}

      dynamicGpuBtn.href = `https://${{amzDomain}}/s?k=${{searchHw}}&tag=${{amzTag}}`;
    }}

    quantSelect.addEventListener('change', recalculateVram);
    ctxSlider.addEventListener('input', recalculateVram);
    gpuTierSelect.addEventListener('change', recalculateVram);
    recalculateVram();
  </script>
</body>
</html>"""

    os.makedirs(DOCS_DIR, exist_ok=True)
    with open(os.path.join(DOCS_DIR, f"{slug}.html"), "w", encoding="utf-8") as f:
        f.write(html)


def rebuild_portal_sitemap_and_rss(published, mon_cfg):
    header_cta = mon_cfg.get("header_cta", {})
    header_cta_url = build_amazon_url(header_cta.get("search_query", "NVIDIA+GeForce+RTX+4090+Graphics+Card"), mon_cfg)

    cards_html = ""
    for item in reversed(published):
        cards_html += f"""
        <a href="{item['slug']}.html" class="block bg-neutral-900 border border-neutral-800 hover:border-lime-500/60 rounded-xl p-5 transition group">
          <div class="flex items-center justify-between mb-2">
            <span class="text-xs font-black uppercase bg-lime-500 text-neutral-950 px-2.5 py-0.5 rounded">Q4_K_M: {item.get('vram_q4', 20)} GB VRAM</span>
            <span class="text-xs text-neutral-400">{item.get('params_b', 32)}B Params</span>
          </div>
          <h2 class="text-base font-bold text-white group-hover:text-lime-400 transition">{item['model_name']} VRAM & GPU Calculator</h2>
          <p class="text-xs text-neutral-400 mt-2 line-clamp-2">{item['description']}</p>
          <span class="inline-block mt-4 text-xs font-extrabold text-lime-400">Calculate Quantization & Token Speed →</span>
        </a>"""

    index_html = f"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>VRAMCalcIndex — Local LLM VRAM, GGUF Quantization & GPU Sizing Calculator (2026)</title>
  <meta name="description" content="Calculate exact GPU VRAM, Apple Unified Memory, and tokens/sec for Llama 3.3, DeepSeek R1, Qwen 2.5, Mistral, and Gemma models.">
  <link rel="canonical" href="{SITE_BASE_URL}/">
  <link rel="alternate" type="application/rss+xml" title="VRAMCalcIndex RSS Feed" href="{SITE_BASE_URL}/rss.xml">
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-neutral-950 text-neutral-100 min-h-screen flex flex-col">
  <header class="border-b border-neutral-800 bg-neutral-900/80 sticky top-0 z-30 backdrop-blur">
    <div class="max-w-5xl mx-auto px-4 py-3.5 flex items-center justify-between">
      <div class="text-lg font-black text-white flex items-center gap-2">
        <span class="bg-lime-500 text-neutral-950 px-2 py-0.5 rounded font-black">VRAM</span>CalcIndex
      </div>
      <a href="{header_cta_url}" target="_blank" rel="nofollow sponsored" class="text-xs sm:text-sm bg-lime-500 hover:bg-lime-400 text-neutral-950 font-extrabold px-3.5 py-2 rounded-lg transition">
        {header_cta.get('cta_text', 'Shop Local AI GPUs on Amazon →')}
      </a>
    </div>
  </header>

  <main class="max-w-5xl mx-auto px-4 py-12 flex-1 w-full">
    <div class="text-center max-w-2xl mx-auto mb-10">
      <h1 class="text-3xl sm:text-5xl font-black text-white tracking-tight">Local LLM VRAM & GPU Hardware Sizing Database</h1>
      <p class="text-neutral-400 mt-3 text-base sm:text-lg">Select any open-weights AI model below to calculate exact FP16, Q8_0, and Q4_K_M VRAM requirements, KV-cache overhead, and tokens/sec across NVIDIA RTX and Apple Silicon.</p>
      <div class="mt-6">
        <input id="filterInput" type="text" placeholder="Filter AI model (e.g. Llama 3.3 70B, DeepSeek R1, Qwen 2.5 Coder, Mistral, 24GB VRAM...)" class="w-full bg-neutral-900 border border-neutral-700 rounded-xl px-4 py-3.5 text-sm text-white focus:outline-none focus:border-lime-500">
      </div>
    </div>

    <div id="modelGrid" class="grid sm:grid-cols-2 lg:grid-cols-3 gap-5">{cards_html}</div>
  </main>

  <footer class="border-t border-neutral-800 py-8 text-center text-xs text-neutral-500">
    <p>© {datetime.now(timezone.utc).year} VRAMCalcIndex. As an Amazon Associate, we earn from qualifying purchases.</p>
  </footer>

  <script>
    const filterInput = document.getElementById('filterInput');
    const cards = Array.from(document.getElementById('modelGrid').children);
    filterInput.addEventListener('input', (e) => {{
      const q = e.target.value.toLowerCase();
      cards.forEach(c => {{
        c.style.display = c.textContent.toLowerCase().includes(q) ? 'block' : 'none';
      }});
    }});
  </script>
</body>
</html>"""

    os.makedirs(DOCS_DIR, exist_ok=True)
    with open(os.path.join(DOCS_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(index_html)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    urls_xml = f"""  <url>
    <loc>{SITE_BASE_URL}/</loc>
    <lastmod>{today}</lastmod>
    <changefreq>daily</changefreq>
    <priority>1.0</priority>
  </url>\n"""

    for item in published:
        urls_xml += f"""  <url>
    <loc>{SITE_BASE_URL}/{item['slug']}.html</loc>
    <lastmod>{item.get('date', today)}</lastmod>
    <changefreq>monthly</changefreq>
    <priority>0.9</priority>
  </url>\n"""

    with open(os.path.join(DOCS_DIR, "sitemap.xml"), "w", encoding="utf-8") as f:
        f.write(f"""<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n{urls_xml}</urlset>""")

    rss_items = ""
    for item in list(reversed(published))[:30]:
        rss_items += f"""    <item>
      <title>{item['title']}</title>
      <link>{SITE_BASE_URL}/{item['slug']}.html</link>
      <guid>{SITE_BASE_URL}/{item['slug']}.html</guid>
      <description>{item['description']}</description>
    </item>\n"""

    rss_xml = f"""<?xml version="1.0" encoding="UTF-8" ?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>VRAMCalcIndex Local LLM Hardware Database</title>
    <link>{SITE_BASE_URL}/</link>
    <description>Interactive Local LLM VRAM, GGUF Quantization & GPU Calculators</description>
    <atom:link href="{SITE_BASE_URL}/rss.xml" rel="self" type="application/rss+xml" />
    <atom:link href="https://pubsubhubbub.appspot.com/" rel="hub" />
{rss_items}  </channel>
</rss>"""

    with open(os.path.join(DOCS_DIR, "rss.xml"), "w", encoding="utf-8") as f:
        f.write(rss_xml)

    with open(os.path.join(DOCS_DIR, "robots.txt"), "w", encoding="utf-8") as f:
        f.write(f"User-agent: *\nAllow: /\nSitemap: {SITE_BASE_URL}/sitemap.xml\n")

    with open(os.path.join(DOCS_DIR, f"{INDEXNOW_KEY}.txt"), "w", encoding="utf-8") as f:
        f.write(INDEXNOW_KEY)


def notify_search_engines(new_urls):
    if not SITE_BASE_URL or not new_urls:
        return
    parsed_host = urllib.parse.urlparse(SITE_BASE_URL).netloc
    payload = {
        "host": parsed_host,
        "key": INDEXNOW_KEY,
        "keyLocation": f"{SITE_BASE_URL}/{INDEXNOW_KEY}.txt",
        "urlList": new_urls
    }
    try:
        req = urllib.request.Request(
            "https://api.indexnow.org/indexnow",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            print(f"[INDEXNOW] Pushed {len(new_urls)} URLs (HTTP {resp.status})")
    except Exception as e:
        print(f"[WARN] IndexNow ping notice: {e}")

    try:
        hub_data = urllib.parse.urlencode({
            "hub.mode": "publish",
            "hub.url": f"{SITE_BASE_URL}/rss.xml"
        }).encode("utf-8")
        req_hub = urllib.request.Request(
            "https://pubsubhubbub.appspot.com/",
            data=hub_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST"
        )
        with urllib.request.urlopen(req_hub, timeout=15) as resp_hub:
            print(f"[WEBSUB] Google PubSubHubbub ping status: {resp_hub.status}")
    except Exception as e:
        print(f"[WARN] WebSub ping notice: {e}")


def main():
    published = load_json(PUBLISHED_FILE, [])
    mon_cfg = load_json(MONETIZATION_FILE, {})
    published_slugs = {item["slug"] for item in published}

    candidates = []
    if CUSTOM_MODEL:
        param_match = re.search(r"(\d+(?:\.\d+)?)\s*b\b", CUSTOM_MODEL, flags=re.IGNORECASE)
        p_val = float(param_match.group(1)) if param_match else 32.0
        candidates.append((CUSTOM_MODEL.strip(), p_val, 65536, "Transformer GQA", "Local AI Inference"))

    candidates.extend(SEED_MODELS)
    candidates.extend(discover_google_llm_queries())

    new_urls = []
    count = 0

    for model_tuple in candidates:
        if count >= PAGES_PER_RUN:
            break
        model_name = model_tuple[0]
        slug = slugify(f"{model_name}-vram-requirements-calculator")
        if slug in published_slugs:
            continue

        print(f"[ENGINE] Generating VRAM & quantization calculator for: {model_name}...")
        try:
            data = generate_llm_data(model_tuple)
            build_vram_html_page(data, slug, mon_cfg)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            sp = data.get("specs", {})
            p_b = round(safe_float(sp.get("params_billions", model_tuple[1]), model_tuple[1]), 1)
            q4_gb = round(safe_float(sp.get("vram_q4_km_gb", round(p_b * 0.62 + 1.5, 1)), 20.0), 1)

            published.append({
                "model_name": data.get("model_name", model_name),
                "slug": slug,
                "title": data.get("seo_title", f"{model_name} VRAM Calculator"),
                "description": data.get("meta_description", ""),
                "params_b": p_b,
                "vram_q4": q4_gb,
                "date": today
            })
            published_slugs.add(slug)
            if SITE_BASE_URL:
                new_urls.append(f"{SITE_BASE_URL}/{slug}.html")
            count += 1
            time.sleep(2)
        except Exception as e:
            print(f"[ERROR] Failed building {model_name}: {e}")

    save_json(PUBLISHED_FILE, published)
    rebuild_portal_sitemap_and_rss(published, mon_cfg)
    if new_urls and SITE_BASE_URL:
        new_urls.append(f"{SITE_BASE_URL}/")
        notify_search_engines(new_urls)
    print(f"[DONE] Published {count} new Local LLM VRAM calculator pages.")


if __name__ == "__main__":
    main()
