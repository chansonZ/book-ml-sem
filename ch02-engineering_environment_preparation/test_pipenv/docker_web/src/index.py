from flask import Flask
from flask import Response
app = Flask(__name__)
@app.route("/")
def hello():
    res = Response("Hi，Python")
    res.headers['Content-Type'] = 'text/plain'
    return res
if __name__ == "__main__":
    app.run(host='0.0.0.0')
