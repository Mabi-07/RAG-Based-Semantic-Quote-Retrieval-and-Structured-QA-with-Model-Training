from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from data.quotes_dataset import load_quotes

df = load_quotes()

model = SentenceTransformer("all-MiniLM-L6-v2")

# 🔒 Safe sample size
SAMPLE_SIZE = min(5000, len(df))

train_examples = []

for _, row in df.sample(SAMPLE_SIZE, random_state=42).iterrows():
    train_examples.append(
        InputExample(texts=[f"quotes by {row['author']}", row["quote"]])
    )
    train_examples.append(
        InputExample(texts=[f"quotes about {' '.join(row['tags'])}", row["quote"]])
    )

train_loader = DataLoader(train_examples, batch_size=16, shuffle=True)
train_loss = losses.MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(train_loader, train_loss)],
    epochs=1,
    warmup_steps=100
)

model.save("model/embedding_model")
