import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

model_names = {
    "herbert-klej-cased-v1": {
        "tokenizer": "allegro/herbert-klej-cased-tokenizer-v1", 
        "model": "allegro/herbert-klej-cased-v1",
    },
    "herbert-base-cased": {
        "tokenizer": "allegro/herbert-base-cased", 
        "model": "allegro/herbert-base-cased",
    },
    "herbert-large-cased": {
        "tokenizer": "allegro/herbert-large-cased", 
        "model": "allegro/herbert-large-cased",
    },
}

tokenizer = AutoTokenizer.from_pretrained(model_names["herbert-base-cased"]["tokenizer"])
herbert = AutoModel.from_pretrained(model_names["herbert-base-cased"]["model"]).to(device)

@torch.no_grad()
def herbert_forward(data, batch_size=256):
    embeddings = []
    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i:i+batch_size]
        tokens = tokenizer.batch_encode_plus(
            batch,
            padding="longest",
            add_special_tokens=True,
            return_tensors="pt",
        )

        if torch.cuda.is_available():
            for key in tokens.keys():
                tokens[key] = tokens[key].to(device)

        embeddings.append(herbert(**tokens)['pooler_output'].cpu())
    return torch.cat(embeddings)


import xgboost as xgb

model = xgb.XGBRegressor()

def train(batch_size=256, validate=False):
    places = pd.read_csv('places.csv.gz')
    mask = places['language'] == 'pl'
    places = places[mask]
    X = pd.DataFrame(herbert_forward(list(places['query'])).numpy())
    X['category'] = places['category']
    X['audit_latitude'] = places['audit_latitude']
    X['audit_longitude'] = places['audit_longitude']
    y = places['position']
    model.fit(X, y)
    if validate:
        y_pred = model.predict(X)
        print(mean_squared_error(y, y_pred))


