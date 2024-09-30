import txtai
import numpy as np
import pandas as pd

from huggingface_hub import login

login("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")

np.random.seed(1)

df = pd.read_csv ('train.csv') .dropna()
content = df.content_plain.values

embeddings = txtai.Embeddings({
  'path': 'sentence-transformers/paraphrase-mpnet-base-v2'
})

embeddings.index (content)
embeddings.save('embeddings_train.tar.gz')
