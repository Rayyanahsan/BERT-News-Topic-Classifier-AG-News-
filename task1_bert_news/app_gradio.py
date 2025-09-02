import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

model_ckpt = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt)
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)

def classify(text):
    return pipe(text)

iface = gr.Interface(fn=classify, inputs=gr.Textbox(lines=2, label='Headline'), outputs='json', title='AG News — BERT Classifier')
iface.launch()
