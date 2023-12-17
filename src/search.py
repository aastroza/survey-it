import pandas as pd
import numpy as np
from typing import List

from src.embed import get_embedding
from src.utils import cosine_similarity
from src.config import EMBEDDING_MODEL

# Load Data
df = pd.read_csv('../data/processed/embedded_questions.csv')
df['ada_embedding'] = df.ada_embedding.apply(eval).apply(np.array)

def get_reference_questions(question: str, n: int, threshold: float = None) -> List[str]:

   df_search = df.copy() 
   embedding = get_embedding(question, model=EMBEDDING_MODEL)
   df_search['similarities'] = df_search.ada_embedding.apply(lambda x: cosine_similarity(x, embedding))

   if threshold:
       df_search = df_search[df_search.similarities >= threshold]

   res = (df_search
          .sort_values('similarities', ascending=False)
          .head(n)
          .reset_index(drop=True)
          )
   return res[['isPanel', 'surveyCountry', 'surveyQuestion', 'surveyData']]