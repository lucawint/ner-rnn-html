import json

import requests

HOSTNAME = 'http://localhost'
PORT = 8000
data_lst = []

s = "The 2007 Bowling Green Falcons football team represented Bowling Green State University in the 2007 NCAA football season . The team was coached by Gregg Brandon and played their home games in Doyt Perry Stadium in Bowling Green , Ohio . It was the 89th season of play for the Falcons . Bowling Green finished the season 8-5 overall and has finished 4-2 in the MAC East . They participated in the GMAC Bowl , losing to Tulsa 63-7 . Bowling Green was picked to finish fifth in the MAC East Division by the MAC News Media Association . Three Falcons , Senior Kory Lichtensteiger and Juniors Erique Dozier and Corey Partridge , garnered preseason honors by being named to All-MAC preseason teams . The 2007 Bowling Green Falcons football team consists of 95 total players . The class breakdown of these players is 12 seniors , 21 juniors , 30 sophomores , 12 redshirt freshman , and 18 true freshman . Returning starters from the 2006 team are six offensive starters and eight defensive starters . Overall , 53 lettermen are returning from the 2006 team ( 25 on offense , 28 on defense and 0 on special teams ) . As the 2007 college football season neared the end , many organizations began to announce finalists and winners of various post-season awards . Kory Lichtensteiger was named a finalist for the Rimington Trophy , given to the nation 's best center . He was one of 6 athletes recognized . Bowling Green also had eight players make the All-Conference Teams ( the fourth most of any school in the MAC ) ."

data_lst.append(s)

ner_api = '{hostname}:{port}/ner'.format(hostname=HOSTNAME,
                                         port=PORT)

resp = requests.post(ner_api, json=json.dumps(data_lst))
response_lst = json.loads(resp.text)
for idx, data in enumerate(data_lst):
    print('Original:\t{}'.format(data))
    print('Tagged:\t\t{}'.format(response_lst[idx]))
