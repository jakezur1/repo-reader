import google.generativeai as genai
import gh_api_caller

GOOGLE_API_KEY='AIzaSyD5UWAso7Dmd6XpUp3wds1GA0-GkTWrddg'
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                                system_instruction="You are a github assistant, at the start of each chat you will be given a repo full of code and your job is to answer queries related to that original code base")
#chat = None

def readrepo(code_string: str, chat_history=[]) -> str:

    chat = model.start_chat(history=chat_history)
    
    response = chat.send_message(f"I want you to read through this code and understand the various elements and how they connect to each other and just respond with the word success if you have completed this task: {code_string}")
    return response.text, chat

def chat_func(query_string: str, chat) -> str:
    if chat is None:
        raise ValueError("Chat session is not initialized. Please initialize it by calling readrepo first.")
    
    response = chat.send_message(query_string)
    return response.text
    
def genReadMe(output_full: str, output_req: str, output_sh: str) -> str:
    #model = genai.GenerativeModel('gemini-pro')
    template = read_md_to_string("template_readme/Project.md")
    response = model.generate_content(f"""
                                      Generate a ReadME file for this code using this template {template}.
                                      I need a Title for the project that replaces the 'Project Title' header
                                      and write as if you were the developer of this code base. 
                                      Only use the sections of the template that are relevant to the code: {output_full} 
                                      The code is in the format: (filepath: content) for each file thorugh each directory and subdirectory.
                                      Utilize the orgnization of the files for the usage section with details how different files link to each other.
                                      If {output_req} has information utilize (LIST ALL the required dependencies) it for the prerequisites 
                                      section including how to install all requied dependencies.
                                      If {output_sh} has information utilize it for the usage section and installation section, but the usage section 
                                      should undertand how the files link together and give a description of what files to run to start the entire codebase. 
                                      Be detailed with all parts of the ReadME so that any developer can undsertand!
                                      Fill in any URLs possible with their actual URL links otherwise remove them from the template""")

    return response.text

def read_md_to_string(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
def linking_summary(codebase, file_list):
    print(file_list)
    
"""
def list_of_files(owner, repo):
    owner = owner
    repo = repo
    path = ''

    url = f'https://api.github.com/repos/{owner}/{repo}/contents/{path}'

    files = get_all_files(url, headers)
    output_full = ""
    file_list = []

    for file_path, content in files.items():
        output_full += f"{file_path}: {content}"
        file_list.append(file_path)

    return output_full, file_list
"""

if __name__ == '__main__':
    chat = None
    chat_history = []

    #string = "my_list = [5, 2, 8, 3, 1] my_list.sort() print(my_list)"

    output_full, output_req, output_sh = gh_api_caller.main('jakezur1', 'factorlib')
    #output_full, file_list = gh_api_caller.list_of_files('jakezur1', 'factorlib')

    #print(linking_summary(output_full, file_list))
    print(genReadMe(output_full, output_req, output_sh))

    # idk, chat = readrepo(string, chat_history)
    # idk = chat_func("what does this code do", chat)
    # idk = chat_func("are there other ways to do this same thing but with a for loop", chat)
    # print(chat.history)

