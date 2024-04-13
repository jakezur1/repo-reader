import os
import re
import sqlite3
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import json
import requests
import random
import gemini
app = Flask(__name__)
app.secret_key = 'repo'
chat = None


## read me generation
@app.route('/readme', methods=['GET', 'POST'])
def generation():
    method = request.method
    
    if method == 'POST':
        #query = request.json
        #uery = query['message']
        query = "my_list = [5, 2, 8, 3, 1] my_list.sort() print(my_list)"
        print(query)
        ouput = gemini.genReadMe(query)
        response = {
                "message" : ouput
            }
        print(response)
        return jsonify(response)
    
@app.route('/start_chat', methods=['GET', 'POST'])
def parserepo():
    method = request.method
    
    if method == 'POST':
        #query = request.json
        #uery = query['message']
        query = "my_list = [5, 2, 8, 3, 1] my_list.sort() print(my_list)"
        print(query)

        global chat
        ouput, chat = gemini.readrepo(query)

        response = {
                "message" : ouput
            }
        print(response)
        return jsonify(response)
    
@app.route('/chat', methods=['GET', 'POST'])
def chat_on_code():
    method = request.method
    
    if method == 'POST':
        #query = request.json
        #uery = query['message']
        query = "write this as a for loop"

        global chat
        ouput = gemini.chat_func(query, chat)

        response = {
                "message" : ouput
            }
        print(response)
        return jsonify(response)

if __name__ == '__main__':
    app.debug = True
    app.run()