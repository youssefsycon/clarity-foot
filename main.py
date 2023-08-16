import joblib
import spacy
import uvicorn as uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from corrector import *
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class UserCreate(BaseModel):
    message: str


''' charger le model et le vectorizer'''

clf, vectorizer = joblib.load('saved/model.pkl'), joblib.load('saved/vectorizer.pkl')

'''charger le model NER'''
nlp_ner = spacy.load("./foot/model-best")

'''charger le model et le vectorizer pour l'ajout et la suppression'''
clf_ajout, vectorizer_ajout = joblib.load('saved/ajout/model.pkl'), joblib.load('saved/ajout/vectorizer.pkl')


def intent_recognition_action(message):
    X_test = vectorizer_ajout.transform([message])
    predicted_intent_action = clf_ajout.predict(X_test).tolist()
    return predicted_intent_action


def intent_recognition(message):
    X_test = vectorizer.transform([message])
    predicted_intent = clf.predict(X_test).tolist()
    return predicted_intent


def named_entity_recognition(message):
    doc = nlp_ner(message)
    entities = {}
    for ent in doc.ents:
        entities[ent.label_] = ent.text

    return entities


@app.post("/predict")
async def predict(userCreate: UserCreate):
    # text = correct_spelling_open_ai(userCreate.message)
    #
    # if text.endswith("\n"):
    #     text = text[:-1]
    # print(text)
    text = userCreate.message
    predicted_intent = intent_recognition(text)
    predicted_intent_action = intent_recognition_action(text)
    entities = named_entity_recognition(text)

    # for key in entities:
    #     if key == "SAISON" or key == "PERSON":
    #         continue
    #     try:
    #         entities[key] = correct_spelling_open_ai(entities[key])
    #
    #     except:
    #         return "requet number passed"

    return {"action": predicted_intent_action[0], "intent": predicted_intent[0], "entities": entities}


# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000)

