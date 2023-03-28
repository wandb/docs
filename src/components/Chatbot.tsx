// src/components/Chatbot.tsx
import React, { useState, useEffect, ButtonHTMLAttributes } from 'react';
import axios from 'axios';
import styled from '@emotion/styled';

const ChatbotContainer = styled.div`
  position: fixed;
  bottom: 20px;
  right: 20px;
`;

function ChatbotButton(props: ButtonHTMLAttributes<HTMLButtonElement>): JSX.Element {
    return <button className='btn btn-sm btn-outline-secondary float-end' {...props}>{props.children}</button>;
}

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

function SendButton(props: ButtonHTMLAttributes<HTMLButtonElement>): JSX.Element {
    return <button className='btn btn-sm btn-primary' style={{
        cursor: 'pointer',
        backgroundColor: '#1e88e5',
        color: 'white',
        border: 'none',
        padding: '10px',
        fontSize: '16px',
    }} {...props}>{props.children}</button>;
}

type Message =
    | { type: 'sent', sender: 'user' | 'bot', content: string }
    | { type: 'error', error: Error}
    ;


const Chatbot: React.FC = () => {
    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState<Message[]>([]);
    const [inputText, setInputText] = useState('');
    const [queryState, setQueryState] = useState<
        {type: 'idle'} |
        {type: 'waiting'} |
        {type: 'error', error: Error}
    >({type: 'idle'});
    const messagePaneRef = React.useRef<HTMLDivElement>(null);

    React.useEffect(() => {
        const chatContainer = messagePaneRef.current;
        if (!chatContainer) return;
        if (chatContainer.scrollTop + chatContainer.clientHeight !== chatContainer.scrollHeight) {
            chatContainer.scrollTo({behavior: 'smooth', top: chatContainer.scrollHeight});
        }
    }, [messages]);

    const addMessage = (msg: Message) => {
        setMessages((prevMessages) => [...prevMessages, msg]);
    }

    const handleSendMessage = async () => {
        if (queryState.type === 'waiting') return;
        if (!inputText.trim()) return;

        addMessage({ type: 'sent', sender: 'user', content: inputText });
        setInputText('');

        const responseMsg: Message = await (async () => {
            try {
                setQueryState({type: 'waiting'});

                // const response = await axios.post('http://localhost:8000/chat', {
                //     query: inputText,
                // });

                const response = await (async () => {
                    // fake latency
                    await new Promise((resolve) => setTimeout(resolve, 500));

                    if (Math.random() < 0.5) {
                        return {data: {response: '(fake response)'}};
                    } else {
                        throw new Error('(fake error)');
                    }
                })();

                console.log(response)

                setQueryState({type: 'idle'});
                return {type: 'sent', sender: 'bot', content: response.data.response};
            } catch (error) {
                console.error('Error fetching API:', error);
                setQueryState({type: 'error', error});
                return {type: 'error', error};
            }
        })();

        addMessage(responseMsg);
    };

    const waiting = queryState.type === 'waiting';

    return (
        <ChatbotContainer>
            {isOpen ? (
                <ChatWindow>
                    <MessagesContainer ref={messagePaneRef}>
                        {messages.map((message, index) => {
                            switch (message.type) {
                                case 'error':
                                    return <Message key={index} style={{ color: 'red' }}>Error: {message.error.message}</Message>
                                case 'sent':
                                    return <Message key={index}>
                                        <strong>{message.sender}:</strong> {message.content}
                                    </Message>;
                            }
                            const _exhaustiveCheck: never = message;
                        })}
                    </MessagesContainer>
                    <InputContainer>
                        <InputField
                            type="text"
                            style={{ color:'black', backgroundColor: 'white' }}
                            placeholder="Type your message..."
                            value={inputText}
                            onChange={(e) => setInputText(e.target.value)}
                            onKeyPress={(e) => {
                                if (e.key === 'Enter') {
                                    handleSendMessage();
                                }
                            }}
                        />
                        <SendButton
                            onClick={handleSendMessage}
                            disabled={waiting}
                            style={{ position: 'relative' }}
                        >
                            <div role="status" style={{
                                position: 'absolute',
                                margin:'auto',
                                left: 0,
                                width: '100%',
                                textAlign: 'center',
                                visibility: waiting ? 'visible' : 'hidden',
                                }}>...</div>
                            <div style={{visibility: waiting ? 'hidden' : 'visible'}}>Send</div>
                        </SendButton>
                    </InputContainer>
                </ChatWindow>
            ) : (
                <ChatbotButton onClick={() => setIsOpen(true)}>Open Chatbot</ChatbotButton>
            )}
            {isOpen && (
                <ChatbotButton
                    onClick={() => {
                        setIsOpen(false);
                        setMessages([]);
                    }}
                >
                    Close Chatbot
                </ChatbotButton>
            )}
        </ChatbotContainer>
    );
};

export default Chatbot;