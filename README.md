# home-generative-agent

[![GitHub Release][releases-shield]][releases]
[![GitHub Activity][commits-shield]][commits]
[![License][license-shield]](LICENSE)

![Project Maintenance][maintenance-shield]

This project aims to create a generative AI agent integration for Home Assistant (HA) that understands your home's context, learns your preferences, and interacts with you and your home to accomplish activities you find valuable. Built using LangChain and LangGraph, which allow for scalable and cutting-edge solutions by leveraging these world-class frameworks, this project tightly integrates a generative agent into Home Assistant. Using a hybrid cloud-edge solution not only balances cost, accuracy, and latency but also provides the benefits of both cloud and edge computing, such as scalability, real-time processing, and cost-effectiveness.

The following features are supported.

- Create complex Home Assistant automation.
- Image scene analysis and understanding.
- Home state analysis of entities, devices, and areas.
- Full agent control of allowed entities in the home.
- Short- and long-term memory using semantic search.
- Automatic summarization of home state to manage LLM context length.

This integration will set up the `conversation` platform, a convenient feature allowing users to converse directly with the Home Generative Assistant.

## Example Use Cases
### Create an automation.
![Alt text](./assets/automation1.png)

### Create an automation that runs periodically.
![Alt text](./assets/cat_automation.png)

The snippet below shows that the agent is fluent in yaml based on what it generated and registered as an HA automation.

```yaml
alias: Check Litter Box Waste Drawer
triggers:
  - minutes: /30
    trigger: time_pattern
conditions:
  - condition: numeric_state
    entity_id: sensor.litter_robot_4_waste_drawer
    above: 90
actions:
  - data:
      message: The Litter Box waste drawer is more than 90% full!
    action: notify.notify
```

### Check a single camera.
![Alt text](./assets/one_camera.png)

### Check multiple cameras.
https://github.com/user-attachments/assets/230baae5-8702-4375-a3f0-ffa981ee66a3

### Check the history of a light.
![Alt text](./assets/history1.png)

### Report the power consumption of an appliance.
TBA

### Summarize home state.
https://github.com/user-attachments/assets/96f834a8-58cc-4bd9-a899-4604c1103a98

You can create an automation of the home state summary that runs periodically from the HA Blueprint `hga_summary.yaml` located in the `blueprints` folder.

### Long-term memory with semantic search.
![Alt text](./assets/semantic1.png)
![Alt text](./assets/semantic2.png)
![Alt text](./assets/semantic3.png)

You can see that the agent correctly generates the automation below.
```yaml
alias: Prepare Home for Arrival
description: Turn on front porch light and unlock garage door lock at 7:30 PM
mode: single
triggers:
 - at: "19:30:00"
    trigger: time
actions:
  - target:
      entity_id: light.front_porch_light
    action: light.turn_on
    data: {}
  - target:
      entity_id: lock.garage_door_lock
    action: lock.unlock
    data: {}
```

### Check a camera for packages.
![Alt text](./assets/check-for-boxes.png)

Below the camera image the agent analyzed, you can see that two packages are visible. 

![Alt text](./assets/check-for-boxes-pic.png)

### Proactive notification of package delivery.
![Alt text](./assets/proactive-camera-automation.png)

Below is an example notification from this automation if any boxes or packages are visible.

![Alt text](./assets/proactive-notification.png)

## Architecture and Design

Below is a high-level view of the architecture.

![Alt text](./assets/hga_arch.png)

The general integration architecture follows the best practices as described in [Home Assistant Core](https://developers.home-assistant.io/docs/development_index/) and is compliant with [Home Assistant Community Store](https://www.hacs.xyz/) (HACS) publishing requirements.

The agent is built using LangGraph and uses the HA `conversation` component to interact with the user. The agent uses the Home Assistant LLM API to fetch the state of the home and understand the HA native tools it has at its disposal. I implemented all other tools available to the agent using LangChain. The agent employs several LLMs, a large and very accurate primary model for high-level reasoning, smaller specialized helper models for camera image analysis, primary model context summarization, and embedding generation for long-term semantic search. The primary model is cloud-based, and the helper models are edge-based and run under the [Ollama](https://ollama.com/) framework on a computer located in the home. The models currently being used are summarized below.

Model | Location | Purpose
-- | -- | -- |
[GPT-4o](https://platform.openai.com/docs/models#gpt-4o) | OpenAI Cloud | High-level reasoning and planning
[llama-3.2-vision-11b](https://ollama.com/library/llama3.2-vision) | Ollama Edge | Image scene analysis
[llama-3.2-vision-11b](https://ollama.com/library/llama3.2-vision) | Ollama Edge | Primary model context summarization
[mxbai-embed-large](https://ollama.com/library/mxbai-embed-large) | Ollama Edge | Embedding generation for sematic search

### LangGraph-based Agent
LangGraph powers the conversation agent, enabling you to create stateful, multi-actor applications utilizing LLMs as quickly as possible. It extends LangChain's capabilities, introducing the ability to create and manage cyclical graphs essential for developing complex agent runtimes. A graph models the agent workflow, as seen in the image below.

![Alt text](./assets/graph.png)

The agent workflow has five nodes, each Python module modifying the agent's state, a shared data structure. The edges between the nodes represent the allowed transitions between them, with solid lines unconditional and dashed lines conditional. Nodes do the work, and edges tell what to do next.

The ```__start__``` and ```__end__``` nodes inform the graph where to start and stop. The ```agent``` node runs the primary LLM, and if it decides to use a tool, the ```action``` node runs the tool and then returns control to the ```agent```. The ```summarize_and_trim``` node processes the LLM's context to manage growth while maintaining accuracy if ```agent``` has no tool to call and the number of messages meets the below-mentioned conditions.

### LLM Context Management
You need to carefully manage the context length of LLMs to balance cost, accuracy, and latency and avoid triggering rate limits such as OpenAI's Tokens per Minute restriction. The system controls the context length of the primary model in two ways: it trims the messages in the context if they exceed a max parameter, and the context is summarized once the number of messages exceeds another parameter. These parameters are configurable in `const.py`; their description is below.

Parameter | Descritption | Default
-- | -- | -- |
`CONTEXT_MAX_MESSAGES` |  Messages to keep in context before deletion | 100
`CONTEXT_SUMMARIZE_THRESHOLD` | Messages in context before summary generation | 20

The `summerize_and_trim` node in the graph may trim the messages only after content summarization.

### Latency
The latency between user requests or the agent taking timely action on the user's behalf is critical for you to consider in the design. I used several techniques to reduce latency, including using specialized, smaller helper LLMs running on the edge and facilitating primary model prompt caching by structuring the prompts to put static content, such as instructions and examples, upfront and variable content, such as user-specific information at the end. These techniques also reduce primary model usage costs considerably.

You can see the typical latency performance in the table below.

Action | Latency (s) | Remark
-- | -- | -- |
HA intents | < 1 | e.g., turn on a light
Analyze camera image | < 3 | initial request
Add automation | < 1 |
Memory operations | < 1 |

### Tools
The agent can use HA tools as specified in the [LLM API](https://developers.home-assistant.io/docs/core/llm/) and other tools built in the LangChain framework as defined in `tools.py`. Additionally, you can extend the LLM API with tools of your own as well. The code gives the primary LLM the list of tools it can call, along with instructions on using them in its system message and in the docstring of the tool's Python function definition. If the agent decides to use a tool, the LangGraph node `action` is entered, and the node's code runs the tool. The node uses a simple error recovery mechanism that will ask the agent to try calling the tool again with corrected parameters in the event of making a mistake.

The LLM API instructs the agent always to call tools using HA [built-in intents](https://developers.home-assistant.io/docs/intent_builtin) when controlling Home Assistant and to use the intents `HassTurnOn` to lock and `HassTurnOff` to unlock a lock. An intent describes a user's intention generated by user actions.

You can see the list of LangChain tools that the agent can use in the table below.

Langchain Tool | Purpose
-- | -- |
`get_and_analyze_camera_image` | run scene analysis on the image from a camera
`upsert_memory` | add or update a memory
`add_automation` | create and register a HA automation
`get_entity_history` | query HA database for entity history

### Hardware
I built the HA installation on a Raspberry Pi 5 with SSD storage, Zigbee, and LAN connectivity. I deployed the edge models under Ollama on an Ubuntu-based server with an AMD 64-bit 3.4 GHz CPU, Nvidia 3090 GPU, and 64 GB system RAM. The server is on the same LAN as the Raspberry Pi.

## Installation

1. Using the tool of choice, open your HA configuration's directory (folder) (where you find `configuration.yaml`).
2. If you do not have a `custom_components` directory (folder), you must create it.
3. In the `custom_components` directory (folder), create a new folder called `home_generative_agent`.
4. Download _all_ the files from the `custom_components/home_generative_agent/` directory (folder) in this repository.
4. Place the files you downloaded in the new directory (folder) you created.
6. Restart Home Assistant
7. In the HA UI, go to "Configuration" -> "Integrations" click "+," and search for "Home Generative Agent"

## Configuration
Configuration is done in the UI and via the parameters in `const.py`.

<!---->

## Contributions are welcome!

If you want to contribute to this, please read the [Contribution guidelines](CONTRIBUTING.md)

***

[home_generative_agent]: https://github.com/goruck/home-generative-agent
[commits-shield]: https://img.shields.io/github/commit-activity/y/goruck/home-generative-agent.svg?style=for-the-badge
[commits]: https://github.com/goruck/home-generative-agent/commits/main
[forum-shield]: https://img.shields.io/badge/community-forum-brightgreen.svg?style=for-the-badge
[forum]: https://community.home-assistant.io/
[license-shield]: https://img.shields.io/github/license/ludeeus/integration_blueprint.svg?style=for-the-badge
[maintenance-shield]: https://img.shields.io/badge/maintainer-Lindo%20St%20Angel%20%40goruck-blue.svg?style=for-the-badge
[releases-shield]: https://img.shields.io/github/release/goruck/home_generative_agent.svg?style=for-the-badge
[releases]: https://github.com/goruck/home-generative-agent/releases