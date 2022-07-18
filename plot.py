from speech import speech2text
from qa import ask_question
import plotly.express as px
import pandas as pd
from pprint import pprint


QUESTIONS = {
    "title": "What is the best title for this chart described in the text?",
    "type": "What type would you choose for this chart described in the text?",
    "x-axis": "What is plotted on the horizontal x-axis of this chart described in the text?",
    "y-axis": "What is plotted on the vertical y-axis of this chart described in the text?",
    "x-axis-range": "What is the range of values on the horizontal x-axis for this chart described in the text?",
    "y-axis-range": "What is the range of values (from to) on the vertical y-axis for this chart described in the text?",
    "colors": "What are the best colors for this chart described in the text?",

    "mean": "What is the average or mean value of the data of this chart described in the text?",
    "max": "What is the maximum or highest value of the data of this chart described in the text?",
    "min": "What is the minimum or lowest value of the data of this chart described in the text?",
    "correlation": "What type of correlation or when does the data increase or decrease in this chart described in the text?",

    "trends": "What kind of trends, patterns, exceptions or concepts could be found on this chart described in the text?",

    "context": "What kind of domain-specific insights, current events, social and political context, explanations related to this chart described in the text?"
}


def parse_answers(answers):
    tokens = {}
    tokens['title'] = answers['title']
    tokens['type'] = answers['type'].lower()
    tokens['x'] = answers['x-axis']
    tokens['y'] = answers['y-axis']
    if "white" in answers['colors'].lower():
        tokens['color'] = "white"
    elif "brown" in answers['colors'].lower():
        tokens['color'] = "brown"
    else:
        tokens['color'] = "blue"
    if 'increase' in answers['correlation'] or 'increase' in answers['trends']:
        tokens['data'] = pd.DataFrame({tokens['x']: answers['x-axis-range'].split(','), tokens['y']: [5, 10, 20]})
    else:
        tokens['data'] = pd.DataFrame({tokens['x']: answers['x-axis-range'].split(','), tokens['y']: [20, 10, 5]})
    return tokens


def plot(text):
    answers = {}
    for k, q in QUESTIONS.items():
        answers[k] = ask_question(text, q)['answer']
    pprint(answers)
    tokens = parse_answers(answers)
    pprint(tokens)
    if "bar" in tokens['type']:
        fig = px.bar(tokens['data'], x=tokens['x'], y=tokens['y'], title=tokens['title'])
        fig.update_traces(marker_color=tokens['color'])
        fig.show()
    else:
        print("Chart type not recognized")
    return None


if __name__ == "__main__":
    # t = "This is a vertical bar chart entitled “COVID-19 mortality rate by age” that plots Mortality rate by Age. Mortality rate is plotted on the vertical y-axis from 0 to 15%. Age is plotted on the horizontal x-axis in brown bins: 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70-79, 80+. The highest COVID-19 mortality rate is in the 80+ age range, while the lowest mortality rate is in 10-19, 20-29, 30-39, sharing the same rate. The mortality rate increases with age, especially around 40-49 years and upwards. This relates to people’s decrease in their immunity and the increase of co-morbidity with age. The mortality rate increases exponentially with older people. There is no difference in the mortality rate in the range between the age of 10 and 39. The range of ages between 60 and 80+ are more affected by COVID-19. We can observe that the mortality rate is higher starting at 50 years old due to many complications prior. As we decrease the age, we also decrease the values in mortality by a lot, almost to none."
    t = speech2text("sample.flac")
    pprint(t)
    plot(t)
