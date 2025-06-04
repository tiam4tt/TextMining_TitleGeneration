import evaluate
import pandas as pd

# predictions = pd.read_csv("./output/gpt-4o-mini-output.csv").reset_index(drop=True)[
#     "title"
# ]

predictions = pd.read_csv("./output/gemini-2.0-flash-001-output.csv").reset_index(
    drop=True
)["title"]


references = pd.read_csv("./test.csv").reset_index(drop=True)["abstract"]


rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

rouge_scores = rouge.compute(predictions=predictions, references=references)
bert_scores = bertscore.compute(
    predictions=predictions, references=references, lang="en"
)


print("ROUGE:", rouge_scores)
print("BERTScore (averaged):")
print("  Precision:", sum(bert_scores["precision"]) / len(bert_scores["precision"]))
print("  Recall:", sum(bert_scores["recall"]) / len(bert_scores["recall"]))
print("  F1:", sum(bert_scores["f1"]) / len(bert_scores["f1"]))
