# Define negation words
negation_words = {"not", "no", "never", "none", "nothing", "nowhere", "neither", "nor",
                  "can't", "won't", "doesn't", "isn't", "aren't", "wasn't", "weren't",
                  "haven't", "hasn't", "hadn't", "wouldn't", "shouldn't", "mustn't"}

# Function to check if a sentence contains negation words
def contains_negation(sentence):
    words = set(sentence.lower().split())
    return any(word in negation_words for word in words)

# Function to compute token-level overlap between hypothesis and premise
def compute_overlap(premise, hypothesis):
    premise_tokens = set(premise.lower().split())
    hypothesis_tokens = hypothesis.lower().split()
    if not hypothesis_tokens:
        return 0
    overlap_ratio = sum(1 for token in hypothesis_tokens if token in premise_tokens) / len(hypothesis_tokens)
    return overlap_ratio

def get_example_group_results(dataset):
    # Initialise data structures to store results
    results = {
        "Contradiction with High Word Overlap": [],
        "Entailment with High Word Overlap": [],
    }

    # Filter indices for each group
    for idx, example in enumerate(dataset["train"]):
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        label = example["label"]

        overlap_ratio = compute_overlap(premise, hypothesis)
        high_overlap = overlap_ratio >= 0.8  # Threshold set to 80%

        if label == 2 and high_overlap:  # Contradiction with High Word Overlap
            results["Contradiction with High Word Overlap"].append(idx)
        elif label == 0 and high_overlap:  # Entailment with High Word Overlap
            results["Entailment with High Word Overlap"].append(idx)

    return results