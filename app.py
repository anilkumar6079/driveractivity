from flask import Flask, send_file
from flask_restful import Resource, Api, reqparse
import werkzeug
from werkzeug.wsgi import responder

from predict_distracted_video import analyze_video
app = Flask(__name__,static_folder='output')
api = Api(app)


class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

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
            output_path = "./output/"+image_file.filename
            analyze_video(input_path,output_path)
            # response = send_file(output_path)
            return {"path":"/output/"+image_file.filename}
        except Exception as e:
            print("An exception occurred")
            print(e.message)
            return {"status" : "failed","message" : e.message}


api.add_resource(HelloWorld, '/')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
