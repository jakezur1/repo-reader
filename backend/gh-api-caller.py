import os
import sys
import requests
import json

owner = 'jakezur1'
repo = 'factorlib'
path = ''
url = f'https://api.github.com/repos/{owner}/{repo}/contents/{path}'

token = ''
headers = {
    'Accept': 'application/vnd.github.v3.raw+json',
    'Authorization': f'Bearer {token}',
    'X-GitHub-Api-Version': '2022-11-28'
}

def fetch_contents(url, headers):
    response = requests.get(url, headers=headers)
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
            if item['type'] == 'file':
                file_content = requests.get(item['download_url'], headers=headers).text
                all_files[item['path']] = file_content
            elif item['type'] == 'dir':
                all_files.update(get_all_files(item['url'], headers))
    return all_files

def main():
    files = get_all_files(url, headers)
    for file_path, content in files.items():
        print(f"{file_path}: {content}")

if __name__ == "__main__":
    main()

