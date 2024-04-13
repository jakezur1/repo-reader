import os
import sys
import requests
import json


def fetch_contents(url, headers):
    response = requests.get(url, headers=headers)
    print(response)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data: {response.status_code}")
        return None

def get_all_files(url, headers):
    items = fetch_contents(url, headers)
    all_files = {}
    if items:
        for item in items:
            if item['type'] == 'file' and ((".ipynb" not in item['name']) and (".md" not in item['name']) and (".gitignore" not in item['name'])):
                #print(item['name'])
                file_content = requests.get(item['download_url'], headers=headers).text
                all_files[item['path']] = file_content
            elif item['type'] == 'dir':
                all_files.update(get_all_files(item['url'], headers))

    #print(all_files)
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

def main(owner, repo):
    owner = owner
    repo = repo
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
        'Authorization': f'Bearer {token}', 
        'X-GitHub-Api-Version': '2022-11-28'
    }

    files = get_all_files(url, headers)
    output_full = ""
    output_req = ""
    output_sh = ""

    # commits
    get_commit_history(commit_url, commit_headers)
    # ary do what u like here with commits

    for file_path, content in files.items():
        #print(f"{file_path}: {content}")
        if "requirements.txt" in file_path:
            output_req += f"{file_path}: {content}"
        elif ".sh" in file_path:
            output_sh += f"{file_path}: {content}"
        
        output_full += f"{file_path}: {content}"

    return output_full, output_req, output_sh

if __name__ == "__main__":
    output_full, output_req, output_sh = main()
    #print(output_full)
    #print(output_req)
