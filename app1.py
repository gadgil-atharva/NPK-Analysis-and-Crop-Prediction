import firebase_admin
from firebase_admin import db, credentials

cred = credentials.Certificate("credentials.json")
firebase_admin.initialize_app(cred,{"databaseURL": "https://esp8266mliot-default-rtdb.asia-southeast1.firebasedatabase.app/"})

ref1 = db.reference("/N")
N = ref1.get()
ref2 = db.reference("/P")
P = ref2.get()
ref3 = db.reference("/K")
K = ref3.get()
print(N,P,K)