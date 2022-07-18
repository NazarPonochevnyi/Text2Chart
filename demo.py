import gradio as gr
from speech import speech2text
from gptj import get_code
from qa import ask_question
from plot import QUESTIONS
import plotly.express as px
from pprint import pprint


def asr(audio):
    text = speech2text(audio)
    print()
    pprint(f"Text: {text}")
    return text


def chart_generation(text):
    cmd = get_code(text)
    i = cmd.find("], l")
    if "0.6, 0.7" in cmd and i >= 0:
        cmd = cmd[:i] + ", 0.8" + cmd[i:]
    pprint(f"Code: {cmd}")
    try:
        fig = eval(cmd).update_layout(margin=dict(b=50))
    except Exception as e:
        fig = px.line(title=f"Exception: {e}. Please try again!")
    if fig.layout.title['text'] == "Our Zoo":
        new_title = ask_question(text, QUESTIONS["title"])["answer"]
        fig.layout.title['text'] = new_title
    fig.write_image("fig.svg")
    return "fig.svg"


voice_recognition = gr.Interface(fn=asr, 
                                 inputs=gr.components.Audio(source="microphone", type="filepath", label="Voice", interactive=True), 
                                 outputs="text")
chart_creation = gr.Interface(fn=chart_generation, 
                              inputs="text", 
                              outputs=gr.components.Image(type="filepath", label="Chart"))

gr.Series(voice_recognition, chart_creation,
        examples=[
            ["./examples/example01.wav"],
            ["./examples/example02.wav"]
        ],
        title="The Voice Interface for Chart Creation", 
        cache_examples=True, 
        description="Describe the chart you are looking for and click Submit!").launch(enable_queue=True, debug=True)
