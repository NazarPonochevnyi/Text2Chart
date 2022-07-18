from transformers import pipeline


model_name = "deepset/roberta-base-squad2"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)


def ask_question(context, question):
    QA_input = {
        'question': question,
        'context': context
    }
    res = nlp(QA_input)
    return res


if __name__ == "__main__":
    c = "The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks."
    q = "Why is model conversion important?"
    print(ask_question(c, q))
