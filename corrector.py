import cohere
import openai

co = cohere.Client('Wc9Ck1heQqQHcFRY13gbvL8v3RJqdoOgo0CuLw8P')


def correct_spelling(text):
    response = co.generate(
        model='4104831d-da9b-40f7-b9a3-b28394046b54-ft',
        prompt=text)

    return response.generations[0].text


def correct_spelling_open_ai(text):
    openai.api_key = "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"

    response = openai.Edit.create(
        model="text-davinci-edit-001",
        input=text,
        instruction="fix the spelling",
        temperature=0,
        top_p=1
    )
    correct = response.choices[0].text
    return correct


