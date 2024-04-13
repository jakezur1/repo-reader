import os
import re
import sqlite3
from flask import Flask, render_template, request, session, redirect, url_for, jsonify, send_file
import json
import requests
import random
import gemini
import gh_api_caller
app = Flask(__name__)
app.secret_key = 'repo'
chat = None


## read me generation
@app.route('/readme', methods=['GET', 'POST'])
def generation():
    method = request.method
    
    if method == 'POST':
        query = request.json
        
        username = query['username']
        repo = query['repository']
        print(username)
        print(repo)

        #query = "my_list = [5, 2, 8, 3, 1] my_list.sort() print(my_list)"

        full, req, sh = gh_api_caller.main(username, repo)

        ouput = gemini.genReadMe(full, req, sh)
        #ouput = "dfd"

        # Open a text file in append mode
        with open("ReadME.md", "a") as file:
            file.write("")
            file.write(ouput)
        file.close()
        
        # response = {
        #         "message" : ouput
        #     }
        # print(response)
        # return jsonify(response)

        try:
            return send_file('ReadME.md', as_attachment=True)
        except Exception as e:
            return str(e)
    
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