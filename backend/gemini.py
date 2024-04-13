import google.generativeai as genai

GOOGLE_API_KEY='AIzaSyD5UWAso7Dmd6XpUp3wds1GA0-GkTWrddg'
genai.configure(api_key=GOOGLE_API_KEY)

def genReadMe(code_string: str) -> str:
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(f"Generate a ReadME file for this code: {code_string}")
    #print(response.text)

    return response.text

if __name__ == '__main__':
    string = "my_list = [5, 2, 8, 3, 1] my_list.sort() print(my_list)"
    print(genReadMe(string))