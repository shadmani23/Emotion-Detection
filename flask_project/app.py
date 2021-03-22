from flask import Flask
import sqlite3

app = Flask(__name__)

@app.route('/')
def home():
	return "<h1>Hello<h1> "

if __name__=='__main__':
	app.run()