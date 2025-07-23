async function sendMessage() {
  const input = document.getElementById('user-input');
  const chatBox = document.getElementById('chat-box');
  const message = input.value.trim();

  if (!message) return;

  // will be showing users measgaes
  const userMsg = document.createElement('div');
  userMsg.className = 'message user';
  userMsg.textContent = `You: ${message}`;
  chatBox.appendChild(userMsg);

  // will be thinking
  const botThinking = document.createElement('div');
  botThinking.className = 'message bot';
  botThinking.textContent = `Bot: Thinking...`;
  chatBox.appendChild(botThinking);

  input.value = '';
  chatBox.scrollTop = chatBox.scrollHeight;

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message })
    });

    const data = await res.json();

    // give the response (replacing thinking )
    botThinking.textContent = `Bot: ${data.reply}`;
  } catch (err) {
    botThinking.textContent = 'Bot: Error talking to the server.';
    console.error(err);
  }

  chatBox.scrollTop = chatBox.scrollHeight;
}
