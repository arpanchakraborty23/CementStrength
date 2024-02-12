from flask import Flask, render_template, jsonify, request, send_file
from src.exception import CustomException
from src.logger import logging as lg
from src.pipline.train_pipline import TrainPipline
from src.pipline.pradiction_pipline import PradictionPipline
import os,sys


app = Flask(__name__)

@app.route('/')
def home():
    return render_template ('index.html')

@app.route('/train')
def train():
    try:
        train_pipline=TrainPipline()
        train_pipline.run_pipline()

        return jsonify('Train successfull')

    except Exception as e:
        raise CustomException(sys,e)    
            

@app.route('/pradict',methods=['GET','POST'])
def pradict():
    try:
        if request.method=='POST':
            data=dict(request.form.items())
            print(data)
            return jsonify('done')


    except Exception as e:
        raise CustomException(sys,e) 

@app.route('/upload',methods=['GET','POST'])
def upload():
    try:

        if request.method=='POST':
            pradiction_pipline=PradictionPipline(request)
            pradiction_file_detail=pradiction_pipline.run_pipline()

            lg.info('prediction completed. Downloading prediction file.')
            return send_file(pradiction_file_detail.pradiction_file_path,
            download_name=pradiction_file_detail.pradiction_file_name,
            as_attachment=True

            )


        else:
            return render_template('templates/upload_file.upload_file.html')

    except Exception as e:
        lg.info('error occured in upload')        



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug= True)        