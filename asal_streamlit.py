import streamlit as st
import time
from utils.predict_opt import PredictOptions
from models import create_model
from data import  create_predict_data
from utils.metrics import f1_score_func
import torch
import re
import pandas as pd

@st.cache(allow_output_mutation=True)
def initialize_model():
    opt = PredictOptions().parse()   # get training options
    model = create_model(opt)      # create a model 
    model.setup(opt)
    return opt, model
    
def get_dataloader(input):
    dataloader = model.create_dataloader(opt, input, None, randomSample=False)
    return dataloader

def get_color(attr):
    # clip values to prevent CSS errors (Values should be from [-1,1])
    attr = max(-1, min(1, attr))
    if attr > 0:
        red   = 255-int(100 * attr)
        green = 255
        blue  = 255-int(100 * attr)
        
    else:
        red   = 255
        green = 255+int(100 * attr)
        blue  = 255+int(100 * attr)
    return "#{}{}{}".format(('0x%02x'%red).split('x')[-1], ('0x%02x'%green).split('x')[-1],('0x%02x'%blue).split('x')[-1])

def format_word_importances(words, importances):
    tags=[]
    assert len(words) <= len(importances)
    for word, importance in zip(words, importances[: len(words)]):
        color = get_color(importance)
        unwrapped_tag = '\colorbox{{{}}}{{{}}}'.format(color,word.lstrip('#'))
        tags.append(unwrapped_tag)
    return "".join(tags)


st.title('AutoSAL')
st.title('Automated Sentiment Analysis Labeling')
st.subheader('Bak Kallemov, Insight AI Fellowship programm AISV20B')
st.subheader('This is a consulting project for Datasaur.ai')
opt, model = initialize_model()
text_type = st.radio('',('single sentence', 'multi-sentence text', 'upload text file'))
if text_type=='single sentence':
    text = [st.text_input('Input your sentence here','',max_chars=300)]
elif text_type=='multi-sentence text':
    text = re.split('[.\n]',st.text_area('Input your multi-sentence text here (sentence separations are periods \'.\' or Enter',''))
else:
    uploaded_file = st.file_uploader("Choose a text file to upload")
    if uploaded_file is not None:
        text=uploaded_file.getvalue()
        st.write(text)
        text = re.split('[.\n]',text)
    else:
        text=''
        
text=[x.lstrip(' ') for x in text if len(x.lstrip(' '))>0]    

if len(text)>0:
    scores, total_score, attributes, vis_data_ig_records = model.predict(get_dataloader(text))

    threshold = st.slider('Threshold for the class probability (confidence score)',0,100,50)
    
    scores=[(opt.inv_label_dict[x[0]] if x[1]*100> threshold else 'n/a', x[1]*100, t) for x,t in zip(scores,text)]
    
    overall_sentiment = opt.inv_label_dict[total_score.index(max(total_score))]
    #print(scores, "\t overall sentiment is %s"%overall_sentiment)
    #print(attributes)


    st.subheader('Sentiment for the overall text is: %s'%overall_sentiment)
    df = pd.DataFrame(
        scores,
        columns=['sentiment', 'class probablity (confidence) \%','sentences'])
    st.write(df)
    #print(attributes[0][0], attributes[0][1])
    #visualization.visualize_text(vis_data_ig_records[0])

    if st.checkbox('Show word importance using integrated gradient analysis'):


        option = st.selectbox(
            'For which sentence you like to get word importance? Select index based on the above table.',
            range(len(text)))#df['sentences'])
        'You selected:', text[option]
        
        ds=format_word_importances(attributes[option][0], attributes[option][1])
        st.latex('\large'+ds)
        
        wi = pd.DataFrame({
            'tokens': attributes[option][0],
            'attributes (importance [-1;1])': attributes[option][1]
        })
        st.write(wi)


