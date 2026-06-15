import re
import logging
from datasets import load_dataset

HEALTH_KW = re.compile(
    r"\b(disease|disorder|syndrome|virus|bacteria|cancer|tumor|vaccine|vaccination|"
    r"drug|medication|therapy|treatment|clinical|medical|surgery|hospital|health|"
    r"nutrition|diet|vitamin|supplement|diabetes|obesity|hypertension|stroke|"
    r"covid|influenza|flu|hiv|aids|alzheimer|dementia|autism|depression|"
    r"anxiety|asthma|arthritis|cardiovascular|lung|respiratory|immune|allerg|"
    r"infection|epidemic|pandemic|mortality|diagnosis|symptom|pharmaceutical|"
    r"biomedical|neurology|psychiatry|oncology|pediatric|geriatric|poison|"
    r"overdose|injury|mental|birth|pregnancy|infant|toxin)\b",
    re.IGNORECASE,
)


logger = logging.getLogger(__name__)

def load_pubmed_texts(num_abstracts: int) -> list[tuple[str, str]]:
    logger.info("Loading up to %d PubMed abstracts from pubmed_qa...", num_abstracts)

    ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")

    texts = []

    for i, ex in enumerate(ds):
        if len(texts) >= num_abstracts:
            break

        contexts = ex.get("context", {}).get("contexts", [])

        if not contexts:
            continue

        abstract = " ".join(str(s) for s in contexts).strip()

        if len(abstract) < 80:
            continue

        pub_id = ex.get("pubid", i)
        texts.append((abstract, f"pubmed_{pub_id}"))

        if len(texts) % 5000 == 0:
            logger.info("  PubMed: loaded %d abstracts so far...", len(texts))

    logger.info("Loaded %d PubMed abstracts.", len(texts))

    return texts


def load_wikipedia_health_texts(num_articles: int) -> list[tuple[str, str]]:
    logger.info("Streaming Wikipedia to find %d health-related articles...", num_articles)

    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )

    texts = []
    scanned = 0

    for article in ds:
        scanned += 1

        if len(texts) >= num_articles:
            break

        if scanned % 20000 == 0:
            logger.info(
                "  Scanned %d Wikipedia articles, found %d health-related...",
                scanned,
                len(texts),
            )

        title = article.get("title", "")
        body = article.get("text", "")

        if not HEALTH_KW.search(title) and not HEALTH_KW.search(body[:600]):
            continue

        words = body.split()

        if len(words) > 1500:
            body = " ".join(words[:1500])

        if len(body.strip()) < 100:
            continue

        article_id = article.get("id", scanned)
        texts.append((body, f"wiki_{article_id}"))

    logger.info(
        "Loaded %d Wikipedia health articles after scanning %d total.",
        len(texts),
        scanned,
    )

    return texts

