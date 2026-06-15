PROMPT_TEMPLATE = """Rewrite each claim as one short, search-ready sentence. Output only the rewritten sentence — nothing else.

CLAIM: Scientists say vitamin C supplements definitely prevent the common cold in all people.
SIMPLIFIED CLAIM: Vitamin C supplements prevent the common cold.

CLAIM: Researchers found that drinking coffee every morning has been proven to always reduce the risk of heart disease.
SIMPLIFIED CLAIM: Coffee consumption reduces the risk of heart disease.

CLAIM: A new study shows that angioplasty, a procedure to open blocked arteries, can now be performed through the wrist instead of the groin.
SIMPLIFIED CLAIM: Angioplasty can be performed through the wrist.

CLAIM: {claim}
SIMPLIFIED CLAIM:"""
