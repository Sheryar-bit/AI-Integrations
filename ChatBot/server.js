const express = require('express');
require('dotenv').config();
const axios = require('axios');
const app = express();
// const genai = require('@google/gemini-ai');

app.use(express.json());

app.use(express.urlencoded({ extended: true }));
app.use(express.static('public'));

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const GEMINI_URL = process.env.GEMINI_URL;


app.get('/', function(req, res){
    res.sendFile(__dirname + '/public/index.html');
})

app.post('/chat', async (req, res) => {
  const { message } = req.body;

  if (!message) return res.status(400).json({ error: 'Message is required' });

  try {
    const response = await axios.post(
      `${GEMINI_URL}?key=${GEMINI_API_KEY}`,
      {
        contents: [{ parts: [{ text: message }] }]
      },
      {
        headers: {
          'Content-Type': 'application/json'
        }
      }
    );

    const reply = response.data.candidates[0]?.content?.parts[0]?.text || "No response";
    res.json({ reply });
  } catch (err) {
    console.error(err?.response?.data || err.message);
    res.status(500).json({ error: 'Failed to get response from Gemini API' });
  }
});


const PORT = process.env.PORT || 3000;
app.listen(PORT, function() {
    console.log(`Server is running on port ${PORT}`);
})