// src/components/Chatbot.tsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import styled from '@emotion/styled';

const ChatbotContainer = styled.div`
  position: fixed;
  bottom: 20px;
  right: 20px;
`;

const ChatbotButton = styled.button`
  cursor: pointer;
`;

const ChatWindow = styled.div`
  background-color: white;
  border: 1px solid #ccc;
  border-radius: 5px;
  width: 300px;
  height: 400px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
`;

const MessagesContainer = styled.div`
  flex-grow: 1;
  overflow-y: auto;
  padding: 10px;
`;

const Message = styled.div`
  margin-bottom: 10px;
`;

const InputContainer = styled.div`
  display: flex;
  border-top: 1px solid #ccc;
`;

const InputField = styled.input`
  flex-grow: 1;
  background-color: #1e88e5;
  color: white;
  border: none;
  padding: 10px;
  font-size: 16px;
`;

const SendButton = styled.button`
  cursor: pointer;
  background-color: #1e88e5;
  color: white;
  border: none;
  padding: 10px;
  font-size: 16px;
`;

const ClearButton = styled.button`
    cursor: pointer;
    background - color: #f44336;
    color: white;
    border: none;
    padding: 10px;
    font - size: 16px;`
    ;

interface Message {
    sender: 'user' | 'bot';
    content: string;
}

const Chatbot: React.FC = () => {
    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState<Message[]>([]);
    const [inputText, setInputText] = useState('');

    const handleSendMessage = async () => {
        if (!inputText.trim()) return;

        setMessages((prevMessages) => [...prevMessages, { sender: 'user', content: inputText }]);
        setInputText('');

        try {
            const response = await axios.post('http://localhost:8000/chat', {
                query: inputText,
            });
            console.log(response)
            setMessages((prevMessages) => [
                ...prevMessages,
                { sender: 'bot', content: response.data.response },
            ]);
            // setMessages((prevMessages) => [
            //     ...prevMessages,
            //     { sender: 'bot', content: "text" },
            // ]);
        } catch (error) {
            console.error('Error fetching API:', error);
        }
    };


    const handleClearMessages = () => {
        setMessages([]);
    };

    return (
        <ChatbotContainer>
            {isOpen ? (
                <ChatWindow>
                    <MessagesContainer>
                        {messages.map((message, index) => (
                            <Message key={index}>
                                <strong>{message.sender}:</strong> {message.content}
                            </Message>
                        ))}
                    </MessagesContainer>
                    <InputContainer>
                        <InputField
                            type="text"
                            placeholder="Type your message..."
                            value={inputText}
                            onChange={(e) => setInputText(e.target.value)}
                            onKeyPress={(e) => {
                                if (e.key === 'Enter') {
                                    handleSendMessage();
                                }
                            }}
                        />
                        <SendButton onClick={handleSendMessage}>Send</SendButton>
                        <ClearButton onClick={handleClearMessages}>Clear</ClearButton>
                    </InputContainer>
                </ChatWindow>
            ) : (
                <ChatbotButton onClick={() => setIsOpen(true)}>Open Chatbot</ChatbotButton>
            )}
            {isOpen && (
                <ChatbotButton
                    onClick={() => {
                        setIsOpen(false);
                        handleClearMessages();
                    }}
                >
                    Close Chatbot
                </ChatbotButton>
            )}
        </ChatbotContainer>
    );
};

export default Chatbot;