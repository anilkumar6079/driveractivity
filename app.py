from flask import Flask, send_file
from flask_restful import Resource, Api, reqparse
import werkzeug
from werkzeug.wsgi import responder
from flask_cors import CORS
from predict_distracted_video import analyze_video
from predict_distracted_image import  predict_img
app = Flask(__name__,static_folder='output')
api = Api(app)
CORS(app)
import os


class PredictionAPI(Resource):
    def get(self):
        return {'status': 'alive!'}

    def post(self):
        parse = reqparse.RequestParser()
        parse.add_argument(
            'file', type=werkzeug.datastructures.FileStorage, location='files')
        args = parse.parse_args()
        print(args)
        image_file = args['file']
        image_file.save("./uploads/"+image_file.filename)
        # TODO analyze video
        try:
            input_path = "./uploads/"+image_file.filename
            if self.isMP4(input_path):
                output_path = "./output/"+os.path.splitext(image_file.filename)[0]+".mp4"
                labels = analyze_video(input_path,output_path)
                return {"path":"output/"+os.path.splitext(image_file.filename)[0]+".mp4","labels" : labels}
            else:
                return {"label" : predict_img(input_path)}    
            # response = send_file(output_path)
        except Exception as e:
            print("An exception occurred")
            print(e)
            return {"status" : "failed"}

    def isMP4(self,path):
        parts = os.path.splitext(path)
        return (len(parts) == 2 and parts[1].upper() == ".MP4")

api.add_resource(PredictionAPI, '/predict')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
