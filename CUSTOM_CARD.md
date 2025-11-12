# Home Assistant Assist Card

The Home Generative Agent integration now includes a custom Lovelace card that provides a beautiful chat interface for interacting with your Home Assistant Assist agent.

## Features

- **Chat Interface**: Clean, modern chat UI with message bubbles
- **Markdown Support**: Renders responses with full Markdown formatting
- **Tool Call Visualization**: Expandable sections showing tool calls and their results
- **Conversation History**: Maintains context across multiple messages
- **Responsive Design**: Works great on desktop and mobile devices
- **Loading Indicators**: Shows when the assistant is processing your request

## Installation

The custom card is automatically installed with the Home Generative Agent integration. After installing the integration and restarting Home Assistant, follow these steps to add the card to your dashboard:

### Step 1: Verify Resource Registration

The card JavaScript is automatically served at `/home_generative_agent/homeassistant-assist-card.js` when the integration loads.

To verify it's working, navigate to (replace `your-ha-url` with your Home Assistant URL):
```
http://your-ha-url:8123/home_generative_agent/homeassistant-assist-card.js
```

You should see JavaScript code. If you get a 404 error:
1. Verify the integration is installed and loaded
2. Check the Home Assistant logs for frontend registration messages
3. Restart Home Assistant

### Step 2: Register the Resource in Lovelace

1. Go to **Settings** → **Dashboards** → **Resources** (three-dot menu in top right)
2. Click **+ Add Resource**
3. Enter the following details:
   - **URL**: `/home_generative_agent/homeassistant-assist-card.js`
   - **Resource type**: JavaScript Module
4. Click **Create**
5. Refresh your browser (Ctrl+Shift+R or Cmd+Shift+R)

### Step 3: Add the Card to Your Dashboard

1. Open the dashboard where you want to add the card
2. Click **Edit Dashboard** (three-dot menu in top right)
3. Click **+ Add Card**
4. Scroll down and select **Custom: Assist Card** (or search for "homeassistant-assist")
5. Configure the card options (see below)
6. Click **Save**

## Configuration

The card supports the following configuration options:

```yaml
type: custom:homeassistant-assist-card
title: Chat with Home Assistant  # Optional: Custom title
pipeline_id: your_pipeline_id     # Optional: Specific pipeline to use
show_tools: true                   # Optional: Show tool calls (default: true)
placeholder: Ask me anything...    # Optional: Custom input placeholder
```

### Basic Configuration Example

```yaml
type: custom:homeassistant-assist-card
title: Home Assistant
```

### Advanced Configuration Example

```yaml
type: custom:homeassistant-assist-card
title: Home Generative Agent
pipeline_id: 01hx3ygc3k0qyg9bz8a5fz9d6k
show_tools: true
placeholder: How can I help you today?
```

## Usage

1. **Send Messages**: Type your message in the input field and press Enter (or click the send button)
2. **Multi-line Messages**: Hold Shift and press Enter to add line breaks
3. **View Tool Calls**: Click on tool call headers to expand and see input/output details
4. **Conversation Context**: The card maintains conversation history automatically

## Styling

The card automatically adapts to your Home Assistant theme, using:

- Primary colors for user messages
- Secondary colors for assistant messages
- Theme-appropriate text colors and backgrounds
- Consistent with Home Assistant's design language

## Troubleshooting

### Card Not Showing Up

If the custom card doesn't appear in the card picker:

1. Clear your browser cache (Ctrl+Shift+R or Cmd+Shift+R)
2. Verify the resource is registered correctly in Settings → Dashboards → Resources
3. Check the browser console for any JavaScript errors (F12)

### Card Shows Error

If the card displays an error:

1. Ensure Home Generative Agent integration is properly installed and configured
2. Check that you have at least one conversation agent configured in Home Assistant
3. Verify your LLM provider (OpenAI, Ollama, Gemini, or Anthropic) is properly configured

### Tool Calls Not Showing

If tool calls aren't visible:

1. Ensure `show_tools: true` is set in the card configuration
2. Verify the agent is actually calling tools (some responses don't require tools)
3. Check the integration logs for any errors

## Pipeline Configuration

To use a specific pipeline:

1. Go to **Settings** → **Voice Assistants**
2. Create or select a pipeline
3. Copy the pipeline ID from the URL or settings
4. Add it to your card configuration: `pipeline_id: YOUR_PIPELINE_ID`

If no pipeline_id is specified, the card uses the default Home Assistant conversation agent.

## Technical Details

- **Type**: LitElement Web Component
- **Dependencies**: lit, marked.js (for Markdown rendering)
- **Size**: ~69KB (minified)
- **Browser Support**: All modern browsers supporting Web Components

## Development

The card source is available at: https://github.com/lemming1337/homeassistant-assist-card

To build from source:

```bash
git clone https://github.com/lemming1337/homeassistant-assist-card.git
cd homeassistant-assist-card
npm install
npm run build
```

The built file will be in `dist/homeassistant-assist-card.js`.

## Support

For issues related to:
- **The Card UI/Functionality**: Report at https://github.com/lemming1337/homeassistant-assist-card/issues
- **Integration Issues**: Report at https://github.com/goruck/home_generative_agent/issues
