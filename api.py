from flask import Flask, request, jsonify
import json
from utils.predict_opt import PredictOptions
from models import create_model

def initialize_model():
    opt = PredictOptions().parse()   # get training options
    model = create_model(opt)      # create a model 
    model.setup(opt)
    return opt, model
    
def get_dataloader(input_texts):
    dataloader = model.create_dataloader(opt, input_texts, None, randomSample=False)
    return dataloader

if __name__ == '__main__':

    app = Flask(__name__)
    app.config["DEBUG"] = True

    opt, model = initialize_model()

    @app.route('/', methods=['GET'])
    def home():
        return "<h1>AutoSAIL:Automated Sentiment Analysis Intepretable Labeling</h1><p>This site is a prototype API for Datasaur.ai sentiment analysis backend.\n 'Bak Kallemov, Insight AI Fellowship programm AISV20B </p>"

    @app.route('/application/get_prediction', methods=['POST'])
    def get_prediction():
        req_data = request.get_json()
        texts=[]
        ids=[]
        ret=dict()
        data_list=[]

        for sentence_json in req_data['sentences']:
            texts.append(sentence_json['text'])
            ids.append(sentence_json['id'])
        if len(texts)>0:
            scores, total_score, attributes, vis_data_ig_records = model.predict(get_dataloader(texts))
            for i, score in enumerate(scores):
                importances = [round(float(x),2) for x in attributes[i][1]]
                sentence={'id':ids[i], 'tokens':attributes[i][0],'token_importances':importances}
                
                data_list.append({'sentence':sentence, 'sentiment':opt.inv_label_dict[score[0]], 'confidence_score':round(score[1],4)})
        return jsonify({'data':data_list})

    @app.route('/query_sentence', methods=['GET'])
    def query_sentence():
        text = request.args.get('sentence')
        #text = re.split('[.\n]',text)
        scores, total_score, attributes, vis_data_ig_records = model.predict(get_dataloader([text]))
        return '''<h1>{} \t The sentiment is: {} </h1>'''.format(text, opt.inv_label_dict[scores[0][0]])
        
    app.run() 
