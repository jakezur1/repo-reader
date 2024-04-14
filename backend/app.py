import os
import re
import sqlite3
from flask import Flask, render_template, request, session, redirect, url_for, jsonify, send_file, make_response
import json
import requests
import random
import gemini
import gh_api_caller
from flask_cors import CORS


app = Flask(__name__)
cors = CORS(app, support_credentials=True)
app.secret_key = 'repo'
chat = None


def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', '*')
    response.headers.add('Access-Control-Allow-Methods', '*')
    return response


@app.route('/readme', methods=['GET', 'POST', 'OPTIONS'])
def generation():
    method = request.method
    
    if request.method == 'OPTIONS':
        
        return _build_cors_preflight_response()
   
    elif request.method == 'POST':
        query = request.get_json()

        username = query['username']
        repo = query['repository']

        if username is None:
            print("204 Error: Username is null")
        if repo is None:
            print("204 Error: Repository is null")

        full, req, sh = gh_api_caller.main(username, repo)
        
        if full is None:
            print("400 Error: Bad Request, Genmini Call Error")
        else:
            print("200 OK: Move Forward with ReadMe Generations")

        output = gemini.genReadMe(full, req, sh)

        if output is None:
            print("400 Error: Bad Request, Genmini Generation Error")
        else:
            print("200 OK: Read Me Generated")

        response = {
                "message" : output
            }

        return jsonify(response)
    
#### Code Review HTTP REQ

@app.route('/code_review', methods=['GET', 'POST', 'OPTIONS'])
def review():
    method = request.method
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
    
    elif request.method == 'POST':

        query = request.get_json()

        username = query['username']
        repo = query['repository']

        if username is None:
            print("204 Error: Username is null")
        if repo is None:
            print("204 Error: Repository is null")

        full, req, sh = gh_api_caller.main(username, repo)
        
        if full is None:
            print("400 Error: Bad Request, Genmini Call Error")
        else:
            print("200 OK: Move Forward with Review Generations")

        error = True

        while error:
            text, mean, output_arr, pros_arr, cons_arr = gemini.code_review(full)

            #ouput = "dfd"

            if (text is None) or (mean is None) or (output_arr is None) or (pros_arr is None) or (cons_arr is None):
                print("400 Error: Bad Request, Genmini Generation Error")

            else:
                print("200 OK: Send to Frontend")
                error = False

        
        response = {
                "mean" : mean,
                "scores_array" : output_arr,
                "pros_array" : pros_arr,
                "cons_array" : cons_arr,
                "message" : text
            }

        return jsonify(response)



@app.route('/start_chat', methods=['GET', 'POST'])
def parserepo():
    method = request.method
    
    if method == 'POST':
        query = request.json

        username = query['username']
        repo = query['repository']
       
        print(username)
        print(repo)
        full, req, sh = gh_api_caller.main(username, repo)
        
        global chat
        ouput, chat = gemini.readrepo(full)
        return 
    
@app.route('/chat', methods=['GET', 'POST'])
def chat_on_code():
    method = request.method
    
    if method == 'POST':
        query = request.json
        query = query['message']

        global chat
        ouput = gemini.chat_func(query, chat)

        response = {
                "message" : ouput
            }
        
        print(response)
        return jsonify(response)

if __name__ == '__main__':
    app.debug = True
    app.run(debug=True)