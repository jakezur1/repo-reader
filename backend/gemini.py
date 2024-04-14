import google.generativeai as genai
import gh_api_caller
import ast


GOOGLE_API_KEY='AIzaSyD5UWAso7Dmd6XpUp3wds1GA0-GkTWrddg'
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                                system_instruction="You are a github assistant, at the start of each chat you will be given a repo full of code and your job is to answer queries related to that original code base. Utilize concise language")
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
    
# def linking_summary(codebase, file_list):
#     response = model.generate_content(f""" I need a 3-4 sentence description of each file that contains code in this codebase, {codebase} 
#                                             and how they link together using the file list and directory tree, {file_list} """)
    
#     return response.text

def add_comments(codebase):
    response = model.generate_content(f""" Can you add in line comments to the actualy code in cleaner.py that describe each 
                                             how all the key elements of the code work including important functions
                                             and data structures {codebase}""")
    
    return response.text

def code_review(codebase):

    template = read_md_to_string("template_readme/code_review.txt")

    if (codebase is None):
        print("Error 404: Could not parse repositiory, check if repo is public or private")    

    code_metrics = ['Functionality', 'Correctness', 'Code Quality', 'Performance', 'Maintainability', 'Usability']
    # response = model.generate_content(f""" This is an entire code base, {codebase}, including and I need you to tell me what are the most key metrics that can be used to rate the quality of this project, code, doucmentation and codebase overall
    #                                     """)

    response = model.generate_content(f"""  I am giving you to inputs a, an entire codebase:{codebase} and a list of evaluation criteria for this codebase {code_metrics}. 
                                            Evaluate the codebase for each metric in the list from 1 to 10 and 
                                            give a list back of exaclty the same size as the input list where each value corresponds to its input metric. 
                                            Utilize this template {template} to return your information but make sure to follow it exaclty. 
                                            DO NOT change any header that starts with a '#' and make sure to answer all of them as well! KEEP the WHITE space and line spacing the same as well and DO NOT INCLUDE '*' anywhere!
                                            IT IS CURCIAL THAT YOU FOLLOW THE template and the EXAMPLES within the TEMPLATE. IT IS ALSO CRUCIAL that the PRO/CON Bullets are EACH MAX 15 WORDS and THERE IS 2 FOR both Pro and Con.
                                      """)
    
    output_arr = ""
    pros_arr = []
    cons_arr = []
    pros = False
    cons = False
    for line in response.text.splitlines():
        if "Scores:" in line:
            output_arr = line.split(": ",1)[1]
            print(line.split(": ",1)[1])
        
        if pros:
            if ":" in line:
                if "# Cons:" in line:
                    cons = True
                    pros = False
                else:
                    if len(line[(line.find(":") + 2):]) > 5:
                        pros_arr.append(line[(line.find(":") + 2):])
        
        if cons:
            if ":" in line:
                if len(line[(line.find(":") + 2):]) > 5:
                    cons_arr.append(line[(line.find(":") + 2):])

        if "# Pros:" in line:
            pros = True

        if "# Cons:" in line:
            cons = True
            pros = False

        if "Here's an evaluation of the codebase" in line:
            cons = False

    output_arr = ast.literal_eval(output_arr)
    mean = 0
    for i in output_arr:
        mean += i
    mean = round(mean / 6)

    print(mean)
    print(pros_arr)
    print(cons_arr)
    return response.text, mean, output_arr, pros_arr, cons_arr

def test(codebase):
    response = model.generate_content(f""" what are the biggest files in this codebase {codebase}""")
    return response.text



"""
def list_of_files(owner, repo):
    owner = owner
    repo = repo
    path = ''

    url = f'https://api.github.com/repos/{owner}/{repo}/contents/{path}'
    headers = {
        'Accept': 'application/vnd.github.v3.raw+json',
        'Authorization': f'Bearer {token}',
        'X-GitHub-Api-Version': '2022-11-28'
    }

    files = get_all_files(url, headers)
    output_full = ""
    file_list = ""

    for file_path, content in files.items():
        output_full += f"{file_path}: {content}"
        file_list += f"{file_path}, "

    return output_full, file_list
"""

if __name__ == '__main__':
    chat = None
    chat_history = []

    #string = "my_list = [5, 2, 8, 3, 1] my_list.sort() print(my_list)"

    output_full, output_req, output_sh = gh_api_caller.main('jakezur1', 'repo-reader')
    #output_full, file_list = gh_api_caller.list_of_files('jakezur1', 'factorlib')

    print(test(output_full))
    #print(genReadMe(output_full, output_req, output_sh))

    # idk, chat = readrepo(string, chat_history)
    # idk = chat_func("what does this code do", chat)
    # idk = chat_func("are there other ways to do this same thing but with a for loop", chat)
    # print(chat.history)

