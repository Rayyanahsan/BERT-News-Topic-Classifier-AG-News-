import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

st.title('AG News — BERT Classifier')
model_ckpt = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSequenceClassification.from_pretrained('./model') if (st.query_params) else AutoModelForSequenceClassification.from_pretrained(model_ckpt)
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)

txt = st.text_area('Enter a news headline:', 'Stocks rally as market optimism grows')
if st.button('Predict'):
    res = pipe(txt)
    st.write(res)
