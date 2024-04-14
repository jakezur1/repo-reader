# Git. Read. Go.

This project is a Chrome extension that generates ReadMEs and provides repository explanations using Gemini 1.5 Pro, a large language model.

### Features

- Automatically fetches code from public GitHub repositories.
- Utilizes Gemini 1.5 Pro to understand the code's structure and functionality.
- Generates comprehensive ReadME files based on the analyzed code.
- Offers a chat interface to interact with Gemini and ask questions about the repository.

### Prerequisites

To run this application, you will need to install the following dependencies:

*   beautifulsoup4
*   blinker
*   certifi
*   charset-normalizer
*   click
*   Flask
*   geneimpacts
*   google
*   idna
*   inheritance
*   ipython-genutils
*   itsdangerous
*   Jinja2
*   MarkupSafe
*   netifaces
*   openpyxl
*   pickleshare
*   ptyprocess
*   pytz
*   requests
*   simplegeneric
*   soupsieve
*   urllib3
*   wcwidth
*   Werkzeug

You can install these dependencies using pip:

```
pip install -r backend/requirements.txt 
```

## Getting Started

### Installation

1.  Clone the repository:

```
git clone https://github.com/your_username/repo-reader.git
```

2.  Navigate to the project directory:

```
cd repo-reader
```

3.  Install the required NPM packages for the client-side extension:

```
cd client
npm install
```

4.  Build the client-side extension:

```
npm run build
```

5.  Load the unpacked extension in Chrome by navigating to chrome://extensions and enabling Developer mode. 

### Usage

The project consists of two main components: a backend server and a client-side Chrome extension.

**Backend Server:**

The backend server is responsible for interacting with GitHub's API and Gemini 1.5 Pro.

1.  Start the backend server from the root directory:

```
python backend/app.py
```

This will initiate a Flask server running on your local machine, typically at http://127.0.0.1:5000/.

**Client-Side Extension:**

The Chrome extension allows you to interact with the backend server. Once installed, click on the extension icon and choose the desired action.

*   **Generate ReadMe:** Clicking this button will send a message to the background script, which then retrieves the URL of the currently active tab. If the URL belongs to a public GitHub repository, it extracts the username and repository name and sends a POST request to the backend server (http://127.0.0.1:5000/readme). The backend processes the request, generates the ReadME using Gemini, and sends the ReadME back to the client. 
*   **Repo Chat:** (Under Development) This feature will provide a chat interface where you can ask questions about the repository and receive answers from Gemini. 

**File Structure and Interactions:**

Here's a breakdown of the file structure and how different files interact:

*   **backend/app.py:** This is the main Flask application file that defines the server's routes and handles incoming requests. 
*   **backend/gemini.py:** This file contains the functions that interact with the Gemini 1.5 Pro model. It defines functions for initializing the chat, sending messages, and generating ReadMEs.
*   **backend/gh\_api\_caller.py:** This file handles communication with the GitHub API. It fetches the repository's content and extracts relevant information.
*   **client/background.js:** This script runs in the background of the extension and listens for messages from the popup. It also handles communication with the backend server.
*   **client/index.tsx:** The entry point for the client-side extension, rendering the user interface.
*   **client/components/Home.tsx:** This component is responsible for displaying the main interface of the extension, including the buttons for generating ReadMEs and accessing the chat.
*   **client/components/AnimatedTextGradient.tsx:** Creates the animated gradient text effect used in the extension's UI.

## Deployment

This project is designed to run locally as a Chrome extension. 

To deploy the backend server to a live system, you would need to set up a web server (e.g., Apache, Nginx) and configure it to serve the Flask application. For the client-side extension, you would need to package it as a .crx file and publish it to the Chrome Web Store.

### License

This project is licensed under the MIT License.

