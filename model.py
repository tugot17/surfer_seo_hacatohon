import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from xgboost import sklearn
import categories
from sklearn.preprocessing import OneHotEncoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
THRESHOLD = 100

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

model = xgb.XGBRegressor(tree_method='gpu_hist') if torch.cuda.is_available() else xgb.XGBRegressor()

def train(validate=False):
    columns = ['language', 'category', 'query', 'position', 'audit_latitude', 'audit_longitude', 'place_latitude', 'place_longitude']
    places = pd.read_csv('places.csv.gz')[columns].dropna()
    places = places[places['language'] == 'pl'][places['category'].map(places['category'].value_counts()) > THRESHOLD]
    X = pd.DataFrame(herbert_forward(list(places['query'])).numpy())
    # encoder.fit(places['category'].values.reshape(-1,1))
    # cat_df = pd.DataFrame(encoder.transform(places['category'].values.reshape(-1,1)), columns=encoder.categories_[0].tolist())
    # X = pd.concat([X, cat_df], axis=1)
    # X[encoder.categories_[0].tolist()] = cat_df
    places = places.reset_index()
    X['audit_latitude'] = places['audit_latitude']
    X['audit_longitude'] = places['audit_longitude']
    X['place_latitude'] = places['place_latitude']
    X['place_longitude'] = places['place_longitude']
    # X.fillna(len(categories.id_cat))
    y = places['position']
    model.fit(X, y)
    if validate:
        y_pred = model.predict(X)
        print(mean_squared_error(y, y_pred))


def make_query(query_msg, audit_latitude, audit_longitude, place_latitude, place_longitude):
    query = pd.DataFrame(herbert_forward([query_msg]).numpy())
    query['audit_latitude'] = audit_latitude
    query['audit_longitude'] = audit_longitude
    query['place_latitude'] = place_latitude
    query['place_longitude'] = place_longitude
    return model.predict(query)

def save(filename):
    model.save_model(filename)

def init(filename=''):
    if filename:
        model.load_model(filename)
    else:
        train()
