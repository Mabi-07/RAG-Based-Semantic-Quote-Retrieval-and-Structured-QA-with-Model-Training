from datasets import load_dataset
import pandas as pd

def load_quotes():
    dataset = load_dataset("Abirate/english_quotes")
    df = pd.DataFrame(dataset["train"])
    return df[["quote", "author", "tags"]]
