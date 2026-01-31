import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# =========================
# CONFIG
# =========================
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PHASE1_ADAPTER = "phase1_cricket_domain_lora/checkpoint-31000"
PHASE2_ADAPTER = "phase2_cricket_analyst_lora"

DTYPE = torch.float16

# =========================
# TOKENIZER
# =========================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# =========================
# LOAD BASE MODEL (NO DEVICE MAP YET)
# =========================
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=False,   # IMPORTANT
)

# =========================
# LOAD PHASE-1 ADAPTER
# =========================
model = PeftModel.from_pretrained(
    model,
    PHASE1_ADAPTER,
)

# =========================
# LOAD PHASE-2 ADAPTER
# =========================
model = PeftModel.from_pretrained(
    model,
    PHASE2_ADAPTER,
)

# =========================
# NOW APPLY DEVICE MAP + OFFLOADING ONCE
# =========================
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# =========================
# PROMPT
# =========================
def build_prompt(question: str) -> str:
    return f"""
You are a professional cricket analyst.

Answer the question below with clear reasoning and analysis.
Focus on match dynamics, tactics, and cause-effect.
Avoid ball-by-ball commentary and unnecessary statistics.

Question:
{question}

Answer:
""".strip()

# =========================
# GENERATION
# =========================
def ask_analyst(question, max_new_tokens=500):
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# =========================
# TEST PROMPTS
# =========================
EVAL_PROMPTS = [
    "Tell me the best record in England tour of India series. Give me data in the json format only.",
]

# =========================
# RUN
# =========================
if __name__ == "__main__":
    print("\n====== PHASE-2 ANALYST EVALUATION ======\n")

    for q in EVAL_PROMPTS:
        print("Q:", q)
        print("A:")
        print(ask_analyst(q))
        print("\n" + "=" * 60)
