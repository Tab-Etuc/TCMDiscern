"""Active Jeff experiment candidate-count decision.

Frozen BM25 Recall@K records remain unchanged. Primary K=16 fits the
released Jeff A..P decision head in a single forward pass.
"""

PRIMARY_JEFF_K = 16
MAX_DIRECT_JEFF_K = 16
MAX_EVALUATION_K = 35
COMPARISON_KS = (5, 17, 35)
