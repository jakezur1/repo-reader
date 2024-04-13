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



@app.route('/', methods=['GET', 'POST'])

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

if __name__ == '__main__':
    app.debug = True
    app.run()