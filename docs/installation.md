# Installation

- [HACS (Recommended)](#hacs-recommended)
  - [Required Steps](#required-steps)
  - [Optional Add-ons](#optional-add-ons)
- [Manual Install](#manual-install)

---

## HACS (Recommended)

### Required Steps

**1. Install the PostgreSQL with pgvector add-on.**

Click the button below and configure it according to the [add-on documentation](https://github.com/goruck/addon-postgres-pgvector/blob/main/postgres_pgvector/DOCS.md). This provides persistent conversation memory and vector similarity search.

[![Add add-on repository](https://my.home-assistant.io/badges/supervisor_add_addon_repository.svg)](https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https%3A%2F%2Fgithub.com%2Fgoruck%2Faddon-postgres-pgvector)

**2. Install Home Generative Agent from HACS.**

It is available in the default HACS repository, or click the button below to open it directly.

[![Open in HACS](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=goruck&repository=https%3A%2F%2Fgithub.com%2Fgoruck%2Fhome-generative-agent&category=integration)

**3. Restart Home Assistant.**

**4. Add the integration.**

- Go to **Settings → Devices & Services**.
- Click **Add Integration** and search for **Home Generative Agent**.
- Complete the initial instruction-only setup screen.

> If you previously used the legacy single-entry flow, your settings are automatically migrated to the new subentry UI.

**5. Open the integration page and click Setup.**

- Enable the features you want (Conversation, Camera Image Analysis, Conversation Summary are on by default).
- Configure the database connection.
- If no model provider exists yet, you will see a reminder to add one.

**6. Add a Model Provider.**

Click **+ Model Provider** on the integration page and configure at least one provider: OpenAI, Ollama, Gemini, Anthropic, or any OpenAI-compatible endpoint. The first provider is automatically assigned to all features with default models.

**7. Set HGA as your voice assistant.**

- Go to **Settings → Voice Assistants**.
- Select **Home Generative Agent** as the conversation agent.

![Voice assistant config](../assets/hga_assist_config.png)

---

### Optional Add-ons

**Edge model server (Ollama or OpenAI-compatible)**

Run models locally for lower cost and latency.

- Install [Ollama](https://ollama.com/download) on your edge device, **or** run any OpenAI-compatible server (vLLM, llama.cpp, LiteLLM, etc.) and add it as an **OpenAI Compatible** provider.
- Pull the recommended Ollama models:

  ```bash
  ollama pull gpt-oss
  ollama pull qwen3:8b
  ollama pull qwen3-vl:8b
  ollama pull mxbai-embed-large
  ```

**Automation blueprints**

Install the Blueprints from the `blueprints/` directory to create automations that converse directly with the agent. The agent can also create automations for you from conversation without the blueprints. Import each YAML via **Settings → Automations & Scenes → Blueprints** and create automations from the imported blueprints.

**Face recognition**

- Install [face-service](https://github.com/goruck/face-service) on your edge device.
- Enroll people via **Developer Tools → Actions → Enroll Person** in the HA UI.
- To add the enrollment dashboard card, register the Lovelace resource:
  - **Settings → Dashboards → Resources → Add**
  - URL: `/hga-card/hga-enroll-card.js`
  - Type: `JavaScript Module`
- To add the Sentinel proposals card, register this resource as well:
  - URL: `/hga-card/hga-proposals-card.js`
  - Type: `JavaScript Module`

---

## Manual Install

1. Install PostgreSQL with pgvector as shown in step 1 above.
2. Open your HA configuration directory (where `configuration.yaml` lives).
3. Create a `custom_components` directory if it does not already exist.
4. Inside `custom_components`, create a subdirectory named `home_generative_agent`.
5. Download all files from `custom_components/home_generative_agent/` in this repository.
6. Place the downloaded files into the new directory.
7. Restart Home Assistant.
8. Go to **Settings → Devices & Services**, click **Add Integration**, and search for **Home Generative Agent**.
9. Follow steps 4–7 from the [Required Steps](#required-steps) above.
