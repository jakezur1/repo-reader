import os
import sys
import requests
import json

owner = 'jakezur1'
repo = 'factorlib'

def set_variables(own, rep):
    global owner 
    owner = own
    global repo 
    repo = rep
path = ''
url = f'https://api.github.com/repos/{owner}/{repo}/contents/{path}'
commit_url = url = f'https://api.github.com/repos/{owner}/{repo}/contents/commits'

token = ''
headers = {
    'Accept': 'application/vnd.github.v3.raw+json',
    'Authorization': f'Bearer {token}',
    'X-GitHub-Api-Version': '2022-11-28'
}
commit_headers = {
  'Accept': 'application/vnd.github+json',
  'Authorization': 'Bearer {token}', 
  'X-GitHub-Api-Version': '2022-11-28'
}

def fetch_contents(url, headers):
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data: {response.status_code}")
        return None
has_requirements = False
has_sh = False

def get_all_files(url, headers):
    items = fetch_contents(url, headers)
    all_files = {}
    if items:
        for item in items:
            if(item['type'] == 'file' and (".sh" in item['name'])):
                global has_sh
                has_sh = True
            if(item['type'] == 'file' and item['name'] == "requirements.txt"):
                global has_requirements
                has_requirements = True
            if item['type'] == 'file' and ((".ipynb" not in item['name']) and (".md" not in item['name'])):
                print(item['name'])
                file_content = requests.get(item['download_url'], headers=headers).text
                all_files[item['path']] = file_content
            elif item['type'] == 'dir':
                all_files.update(get_all_files(item['url'], headers))

    return all_files

# commit history function, do not call this function with the same url as for repo content
def get_commit_history(url, headers):
    items = fetch_contents(url, headers)
    all_commits = {}
    if items:
        for commit in items:
            commit_sha = commit['sha']
            commit_message = commit['commit']['message']
            commit_author_name = commit['commit']['author']['name']
            commit_author_email = commit['commit']['author']['email']
            commit_date = commit['commit']['author']['date']
            
            all_commits[commit_sha] = {
                'message': commit_message,
                'author_name': commit_author_name,
                'author_email': commit_author_email,
                'date': commit_date    
            }
def main():
    # file content
    files = get_all_files(url, headers)
    output = ""
    for file_path, content in files.items():
        #print(f"{file_path}: {content}")
        output += f"{file_path}: {content}"
        
    # commits


    #print(output)
    #print("Do we have a requirements\n")
    #print(has_requirements)
    return output

if __name__ == "__main__":
    main()

