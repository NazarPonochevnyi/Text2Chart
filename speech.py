from transformers import pipeline#, T5Tokenizer, T5ForConditionalGeneration
import language_tool_python
from deepmultilingualpunctuation import PunctuationModel
from utils import text2int
from nltk import tokenize
from pprint import pprint
import nltk


nltk.download('punkt')
pipe = pipeline("automatic-speech-recognition", "facebook/wav2vec2-large-960h-lv60-self")
# tokenizer = T5Tokenizer.from_pretrained("flexudy/t5-small-wav2vec2-grammar-fixer")
# post_model = T5ForConditionalGeneration.from_pretrained("flexudy/t5-small-wav2vec2-grammar-fixer")
tool = language_tool_python.LanguageToolPublicAPI('en-US')
rpunct = PunctuationModel()


def speech2text(filepath):
    res = pipe(filepath)
    text = res['text']
    # pprint(text)
    matches = tool.check(text)
    text = language_tool_python.utils.correct(text, matches)
    # print(f"Found {len(matches)} mistakes.")
    # pprint(text)
    
    # input_text = "fix: { " + rpunct.restore_punctuation(text) + " } </s>"
    # input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True, add_special_tokens=True)
    # outputs = post_model.generate(
    #     input_ids=input_ids,
    #     max_length=512,
    #     num_beams=4,
    #     repetition_penalty=1.0,
    #     length_penalty=1.0,
    #     early_stopping=True
    # )
    # text = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    text = rpunct.restore_punctuation(text2int(text.lower()))
    # pprint(text)
    text = ' '.join([s.capitalize() for s in tokenize.sent_tokenize(text)])
    # pprint(text)
    return text


if __name__ == "__main__":
    f = "sample.flac"
    print(speech2text(f))
