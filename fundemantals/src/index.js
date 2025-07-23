require('dotenv').config();
const { OpenAI } = require('openai');


const openai = new OpenAI({
  apiKey: process.env.OPENAPI_URL, 
});
const main = async () => {
  const prompt = "How big is Pakistan?";

  const response = await openai.chat.completions.create({
    model: 'gpt-3.5-turbo',
    messages: [
      {
        role: 'user',
        content: prompt,
      },
    ],
    max_tokens: 100,
  });

  console.log(response.choices[0].message.content);
};

main().catch(console.error);
