# Home Assistant Assist Card

The Home Generative Agent integration includes a custom Lovelace card that provides a beautiful chat interface for interacting with your Home Assistant Assist agent.

## Features

- **Chat Interface**: Clean, modern chat UI with message bubbles
- **Markdown Support**: Renders responses with full Markdown formatting
- **Tool Call Visualization**: Expandable sections showing tool calls and their results
- **Conversation History**: Maintains context across multiple messages
- **Responsive Design**: Works great on desktop and mobile devices
- **Loading Indicators**: Shows when the assistant is processing your request

## Installation

The custom card is **automatically downloaded** from the [homeassistant-assist-card repository](https://github.com/lemming1337/homeassistant-assist-card) when the integration loads. After installing the integration and restarting Home Assistant, follow these steps to add the card to your dashboard:

### Step 1: Verify Resource Registration

The card JavaScript is automatically downloaded from GitHub and served at `/home_generative_agent/homeassistant-assist-card.js` when the integration loads.

**Current versions**:
- Card: v0.0.1
- marked.js: v12.0.0

To verify resources are working, navigate to (replace `your-ha-url` with your Home Assistant URL):
```
http://your-ha-url:8123/home_generative_agent/marked.min.js
http://your-ha-url:8123/home_generative_agent/homeassistant-assist-card.js
```

You should see JavaScript code for both. If you get 404 errors:
1. Verify the integration is installed and loaded
2. Check the Home Assistant logs for download and frontend registration messages
3. Ensure your Home Assistant instance can reach GitHub and CDN
4. Restart Home Assistant

### Step 2: Register the Resources in Lovelace

**IMPORTANT**: You need to register TWO resources in the correct order:

#### First: Register marked.js (required dependency)

1. Go to **Settings** → **Dashboards** → **Resources** (three-dot menu in top right)
2. Click **+ Add Resource**
3. Enter the following details:
   - **URL**: `/home_generative_agent/marked.min.js`
   - **Resource type**: JavaScript Module
4. Click **Create**

#### Second: Register the Assist Card

1. Click **+ Add Resource** again
2. Enter the following details:
   - **URL**: `/home_generative_agent/homeassistant-assist-card.js`
   - **Resource type**: JavaScript Module
3. Click **Create**
4. Refresh your browser (Ctrl+Shift+R or Cmd+Shift+R)

**Note**: The order is important - marked.js must be loaded before the assist card.

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

## Automatic Updates

The integration automatically downloads the card from GitHub releases. To update to a newer version:

1. The integration maintainer will update the version number in the code
2. Restart Home Assistant to download the new version
3. Clear your browser cache (Ctrl+Shift+R or Cmd+Shift+R) to load the new version

The downloaded card is cached locally in:
```
config/custom_components/home_generative_agent/www_cache/homeassistant-assist-card/
```

## Troubleshooting

### Card Not Showing Up

If the custom card doesn't appear in the card picker:

1. Clear your browser cache (Ctrl+Shift+R or Cmd+Shift+R)
2. Verify the resource is registered correctly in Settings → Dashboards → Resources
3. Check the browser console for any JavaScript errors (F12)
4. Check Home Assistant logs for download errors: `grep "assist card" home-assistant.log`

### Download Fails

If the integration cannot download the card from GitHub:

1. Verify your Home Assistant instance has internet access
2. Check if GitHub is accessible: `https://github.com/lemming1337/homeassistant-assist-card`
3. Check the Home Assistant logs for specific error messages
4. Ensure no firewall is blocking GitHub access
5. Try restarting Home Assistant to retry the download

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
- **Distribution**: Automatically downloaded from GitHub releases
- **Source Repository**: https://github.com/lemming1337/homeassistant-assist-card
- **Current Version**: v0.0.1

## How It Works

The Home Generative Agent integration uses a modern dependency management approach:

1. **On First Load**: When the integration starts, it checks if the assist card and marked.js are cached locally
2. **Download**: If not cached or if a new version is required, it downloads:
   - **marked.js** from jsDelivr CDN (v12.0.0)
   - **homeassistant-assist-card.js** from GitHub releases (v0.0.1)
3. **Caching**: Both files are cached in `www_cache/homeassistant-assist-card/` for fast subsequent loads
4. **Version Tracking**: Version information is stored to ensure updates are downloaded when available
5. **Static Serving**: The cached files are served via Home Assistant's HTTP server at `/home_generative_agent/`

This approach ensures:
- **Easy Updates**: Update the version numbers and restart to get the latest versions
- **No Manual Installation**: No need to manually copy files or manage dependencies
- **Automatic Verification**: SHA256 checksums are logged for security verification
- **Offline Operation**: Once cached, works offline (until update required)
- **Dependency Management**: marked.js is automatically provided as a required dependency

## Development

The card source is available at: https://github.com/lemming1337/homeassistant-assist-card

To contribute to the card development:

```bash
git clone https://github.com/lemming1337/homeassistant-assist-card.git
cd homeassistant-assist-card
npm install
npm run build
```

The built file will be in the root directory as `homeassistant-assist-card.js`.

### Creating a New Release

To update the card version used by the integration:

1. Create a new release in the homeassistant-assist-card repository
2. Update the `CARD_VERSION` constant in `custom_components/home_generative_agent/frontend.py`
3. Restart Home Assistant to download the new version

## Support

For issues related to:
- **The Card UI/Functionality**: Report at https://github.com/lemming1337/homeassistant-assist-card/issues
- **Integration Issues**: Report at https://github.com/goruck/home_generative_agent/issues
