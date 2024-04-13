import React, { useState, Fragment } from 'react';
import axios from 'axios';
import './RepoChat.css';

// Define the Message type strictly
type Message = {
  content: string;
  sender: 'user' | 'bot';
};

const RepoChat = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');

  const startChat = async () => {
    try {
      const response = await axios.get('http://127.0.0.1:5000/start');
      // Assuming the server sends back a starting message for the chat
      const startMessage: Message = {
        content: response.data.message,
        sender: 'bot'
      };
      setMessages(prevMessages => [...prevMessages, startMessage]);
    } catch (error) {
      console.error('Error starting chat:', error);
    }
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!inputValue.trim()) return;

    // Creating a new user message with a sender type asserted
    const newUserMessage: Message = {
      content: inputValue,
      sender: 'user', // Here sender type is explicitly 'user', matching the Message type
    };
    setMessages(prevMessages => [...prevMessages, newUserMessage]);
    setInputValue('');

    try {
      const response = await axios.post('http://127.0.0.1:5000/chat', { message: inputValue });
      const newBotMessage: Message = {
        content: response.data.message,
        sender: 'bot', // Similarly, ensuring 'bot' is correctly typed
      };
      setMessages(prevMessages => [...prevMessages, newBotMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
    }
  };

  return (
    <div className="repo-chat-container">
      <div className="repo-chat-messages">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.sender}`}>
            {message.content.split('\n').map((line, i) => (
              <Fragment key={i}>
                {line}
                <br />
              </Fragment>
            ))}
          </div>
        ))}
      </div>
      <form className="repo-chat-input-form" onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Type a message..."
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
        />
        <button type="submit">Send</button>
      </form>
      <button onClick={startChat}>Start Chat Here</button>
    </div>
  );
};

export default RepoChat;
