from flask import Flask
from flask_restful import Resource, Api, reqparse
import os
from detect import run
from libxmp.utils import file_to_dict
from novavind import teorcart
from optim_structur import findoptim

stages = [
'site_ready ',
'blades_ready',
"foundation_ready",
"tiers_in_progress"
]


class VEU(Resource):

    def get(self):
        return {'data': "i'm work"}, 200

    def post(self):
        parser = reqparse.RequestParser()
        # parser.add_argument('weights', required=True)
        parser.add_argument('source', required=True)

        args = parser.parse_args()

        # step 1 - detection

        if os.path.exists(args['source']) and os.path.exists('./input/best.pt') \
                and os.path.exists('./input/ves.txt'):
            save_dir = run(source=args['source'], save_txt=True, save_conf=True,
                project='./output', weights='./input/best.pt')

            outdict = {}
            files_in_dir = os.listdir(os.path.join(save_dir, 'labels'))
            for file in files_in_dir:
                with open(os.path.join(save_dir, 'labels', file), 'r') as f:
                    tab = f.readlines()
                    for i, row in enumerate(tab):
                        row_s = row.strip().split(' ')
                        outdict[i] = {'stage': stages[int(row_s[0])],
                                      'x': float(row_s[1]),
                                      'y': float(row_s[2]),
                                      'w': float(row_s[3]),
                                      'h': float(row_s[4]),
                                      'conf': row_s[5]
                                      }

            # step 2 - geo
            xmp = file_to_dict(args['source'])
            xmpdict = {}
            for a, b, _ in xmp['http://www.dji.com/drone-dji/1.0/']:
                xmpdict[a] = b

            if len(xmpdict) == 0:
                return {'data': "i'm dont work. xmp empty"}, 501

            yam = float(xmpdict['drone-dji:FlightYawDegree'])
            pitch = float(xmpdict['drone-dji:FlightPitchDegree'])
            h = float(xmpdict['drone-dji:AbsoluteAltitude'])
            Latitude = float(xmpdict['drone-dji:GpsLatitude'])
            Longitude = float(xmpdict['drone-dji:GpsLongitude'])

            teor_veu = teorcart(yam-90, pitch+90, h/100, (Longitude, Latitude), save_dir)
            row_ind, col_ind = findoptim(teor_veu, outdict, save_dir)
            for x,y in zip(row_ind, col_ind):
                outdict[x]['id'] = teor_veu[y][2]

            return {'data': outdict}, 200
        else:
            return {'data': "i'm dont work. Files not exists"}, 501


app = Flask(__name__)
api = Api(app)

api.add_resource(VEU, '/veu')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # run our Flask app