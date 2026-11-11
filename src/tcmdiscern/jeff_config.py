"""Active Jeff experiment candidate-count decision.

Frozen BM25 Recall@K records remain unchanged. Primary K=16 fits the
released Jeff A..P decision head in a single forward pass.
"""

PRIMARY_JEFF_K = 16
MAX_DIRECT_JEFF_K = 16
MAX_EVALUATION_K = 35
COMPARISON_KS = (5, 17, 35)

# Verified from the published Jeff-1.1-Large model card and pinned jeff_config.json.
JEFF_MODEL_ID = "jgeuter/Jeff-1.1-Large"
JEFF_MODEL_REVISION = "970c21af246a44c5527f7109fca129ab6015c941"
JEFF_UPSTREAM_CODE_TAG = "v1.3.0"
JEFF_BASE_MODEL = "google/gemma-4-31B-it"
JEFF_BASE_REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
JEFF_PROMPT_VERSION = "direct-options-v1"
JEFF_MAX_INPUT_TOKENS = 8192
JEFF_HIDDEN_DIM = 5376
JEFF_HEAD_SOFTCAP = 30.0
JEFF_BIDIRECTIONAL = None  # Jeff-1.1-Large specifically (the other 1.1 models can differ).
JEFF_HEAD_DTYPE = "float32"
JEFF_BACKBONE_DTYPE = "bfloat16"
