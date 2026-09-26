from flask import Flask

app = Flask(__name__)

@approute("/")
def hello():
    return "Hello from flask!"
