from sklearn.model_selection import train_test_split
import pandas as pd
import os
import sys
sys.path.append("../")
from utils import file_util

DATA_DIR = "../data"
DOC_PATH = os.path.join(DATA_DIR, "reuters.csv")
doc_df = pd.read_csv(DOC_PATH, index_col="index")
doc_ids = doc_df.index.values
train_doc, test_doc = train_test_split(doc_ids, test_size=.2, random_state=12)
print("Number of training docs:", len(train_doc))
print("Number of testing docs:", len(test_doc))
train_mask = [False] * len(doc_ids)
test_mask = [False] * len(doc_ids)
for i, did in enumerate(doc_ids):
    if did in train_doc:
        train_mask[i] = True
    else:
        test_mask[i] = True

mask = {"train_mask": train_mask, "test_mask": test_mask}
file_util.dump(mask, os.path.join(DATA_DIR, "mask.pck"))
