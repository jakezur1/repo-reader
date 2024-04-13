import google.generativeai as genai

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
    
def genReadMe(code_string: str) -> str:
    #model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(f"Generate a ReadME file for this code: {code_string}")
    print(response.text)

    return response.text



if __name__ == '__main__':
    chat = None
    chat_history = []

    string = "my_list = [5, 2, 8, 3, 1] my_list.sort() print(my_list)"
    #print(genReadMe(string))

    idk, chat = readrepo(string, chat_history)
    idk = chat_func("what does this code do", chat)
    idk = chat_func("are there other ways to do this same thing but with a for loop", chat)
    print(chat.history)
