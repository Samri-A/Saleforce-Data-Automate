import { useState, useRef, useEffect } from "react";
import { Box, Typography, TextField, IconButton, Paper, Avatar } from "@mui/material";  
import SendIcon from "@mui/icons-material/Send";
import SmartToyIcon from "@mui/icons-material/SmartToy";

async function getAIResponse(userInput: string): Promise<string> {
  const response = await fetch("http://localhost:5000/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ user_input: userInput }),
  });

  if (!response.ok) {
    throw new Error("Failed to fetch AI response");
  }

  const data = await response.json();
  return data.response;
}

function ChatPage() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Hello! I'm your Salesforce Data AI assistant. How can I help you today?",
      sender: "ai",
      time: new Date().toLocaleTimeString(),
    },
  ]);

  const [input, setInput] = useState("");
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = () => {
    if (!input.trim()) return;

    setMessages((prev) => [
      ...prev,
      { id: prev.length + 1, text: input, sender: "user", time: new Date().toLocaleTimeString() },
    ]);
    getAIResponse(input).then((aiResponse) => {
      setMessages((prev) => [
        ...prev,
        { id: prev.length + 1, text: aiResponse, sender: "ai", time: new Date().toLocaleTimeString() },
      ]);
    });
    setInput("");
  };

  return (
    <Box display="flex" flexDirection="column" height="100%">
      <Box 
        flexGrow={1} 
        display="flex" 
        flexDirection="column-reverse" // stack from bottom
        overflow="auto" 
        px={2} 
        py={1}
      >
        <div ref={messagesEndRef} /> {/* scroll anchor */}
        {messages.slice().reverse().map((msg) => (
          <Box
            key={msg.id}
            display="flex"
            justifyContent={msg.sender === "user" ? "flex-end" : "flex-start"}
            mb={1}
          >
            <Paper
              sx={{
                p: 1.5,
                bgcolor: "background.paper",
                color: "text.primary",
                maxWidth: "70%",
                borderRadius: 2,
              }}
            >
              <Box display="flex" alignItems="center">
                {msg.sender === "ai"}
                <Box>
                  <Typography>{msg.text}</Typography>
                  <Typography variant="caption" color="text.secondary">
                    {msg.time}
                  </Typography>
                </Box>
              </Box>
            </Paper>
          </Box>
        ))}
      </Box>

      <Box display="flex" alignItems="center" px={2} py={1}>
        <TextField
          fullWidth
          placeholder="Type your message..."
          variant="outlined"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          sx={{ bgcolor: "background.paper", borderRadius: 1 }}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
        />
        <IconButton color="primary" sx={{ ml: 1 }} onClick={handleSend}>
          <SendIcon />
        </IconButton>
      </Box>
    </Box>
  );
}

export default ChatPage;
