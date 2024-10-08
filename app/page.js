'use client'
import React from 'react';
import { Box, Button, Stack, TextField } from "@mui/material";
import { useState } from "react";

const formatAIResponse = (content) => {
  const lines = content.split('\n');
  let formattedContent = [];
  let currentParagraph = [];

  lines.forEach((line, index) => {
    if (line.trim() === '') {
      if (currentParagraph.length > 0) {
        formattedContent.push(
          <Box key={formattedContent.length} mb={2}>
            {currentParagraph.map((p, i) => (
              <React.Fragment key={i}>
                {p}
                <br />
              </React.Fragment>
            ))}
          </Box>
        );
        currentParagraph = [];
      }
    } else {
      currentParagraph.push(line);
    }
  });

  if (currentParagraph.length > 0) {
    formattedContent.push(
      <Box key={formattedContent.length} mb={2}>
        {currentParagraph.map((p, i) => (
          <React.Fragment key={i}>
            {p}
            <br />
          </React.Fragment>
        ))}
      </Box>
    );
  }

  return formattedContent;
};

export default function Home() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "Hi! I'm the Rate My Professor support assistant. How can I help you today?"
    }
  ]);
  const [message, setMessage] = useState('');

  const sendMessage = async () => {
    setMessages((prevMessages) => [
      ...prevMessages,
      { role: "user", content: message },
      { role: "assistant", content: '' }
    ]);
    setMessage('');

    const response = await fetch('/api/chat', {
      method: "POST",
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify([...messages, { role: "user", content: message }])
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let result = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const text = decoder.decode(value, { stream: true });
      result += text;

      setMessages((prevMessages) => {
        const updatedMessages = [...prevMessages];
        const lastIndex = updatedMessages.length - 1;
        updatedMessages[lastIndex] = {
          ...updatedMessages[lastIndex],
          content: updatedMessages[lastIndex].content + text,
        };
        return updatedMessages;
      });
    }
  };

  return (
    <Box 
      width="100vw" 
      height="100vh" 
      display="flex" 
      flexDirection="column" 
      justifyContent="center" 
      alignItems="center"
    >
      <Stack
        direction="column" 
        width="500px" 
        height="700px"
        border="1px solid black"
        padding={2}
        spacing={3}
      >
        <Box display='flex' justifyContent='center'>Welcome to Rate My Professor</Box>
        <Stack direction="column" spacing={2} flexGrow={1} overflow="auto" maxHeight="100%">
          {messages.map((message, index) => (
            <Box key={index} display="flex" justifyContent={message.role === 'assistant' ? 'flex-start' : 'flex-end'}>
              <Box
                bgcolor={message.role === 'assistant' ? '#4a4a4a' : 'primary.main'}
                color="white"
                borderRadius={16}
                p={3}
              >
                {message.role === 'assistant' ? formatAIResponse(message.content) : message.content}
              </Box>
            </Box>
          ))}
        </Stack>
        <Stack direction="row" spacing={2}>
        <TextField
          label="Message"
          fullWidth
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          sx={{
            '& .MuiOutlinedInput-root': {
              '&.Mui-focused': {
                '& fieldset': {
                  borderColor: '#363636',
                },
              },
            },
            '& .MuiInputLabel-root': {
              '&.Mui-focused': {
                color: '#363636',
              },
            },
          }}
        />

          <Button
            variant="contained"
            sx={{
              bgcolor: '#363636',
              '&:hover': {
                bgcolor: '#6f7070',
              },
            }}
            onClick={sendMessage}
          >
            Send
          </Button>
        </Stack>
      </Stack>
    </Box>
  );
}