# home-generative-agent

[![GitHub Release][releases-shield]][releases]
[![GitHub Activity][commits-shield]][commits]
[![License][license-shield]](LICENSE)

![Project Maintenance][maintenance-shield]

The goal of this project is to create a generative AI agent integration for Home Assistant which allows you to converse with an AI agent that is capable of understanding the state of your home and taking action on your behalf both explicily and proactively. The AI agent is built using [LangChain](https://www.langchain.com/) and [LangGraph](https://www.langchain.com/langgraph) which allows for scalable and cutting edge solutions by leaveragibg these world-class frameworks which are tightly integrated into this Home Assistant integration.

**The follwing features are supported.**

- Create complex Home Assitant automations.
- Image sciene analysis and understanding.
- Home state analysis of entities, devices and areas.
- Full agent control of allowed entities in the home.
- Short- and long-term memory using semantic search.
- Automatic summarization of home state to manage LLM context length.
- A hybrid cloud - edge solution that balances cost, accuracy and latency.

This integration will set up the `conversation` platform which allows the user to
converse with Home Generative Assistant.

## Exaple Use Cases
TBA

## Architecture
TBA

## Installation

1. Using the tool of choice open the directory (folder) for your HA configuration (where you find `configuration.yaml`).
1. If you do not have a `custom_components` directory (folder) there, you need to create it.
1. In the `custom_components` directory (folder) create a new folder called `home_generative_agent`.
1. Download _all_ the files from the `custom_components/home_generative_agent/` directory (folder) in this repository.
1. Place the files you downloaded in the new directory (folder) you created.
1. Restart Home Assistant
1. In the HA UI go to "Configuration" -> "Integrations" click "+" and search for "Home Generative Agent"

## Configuration
Configuration is done in the UI and via the parameters in `constant.py`.

<!---->

## Contributions are welcome!

If you want to contribute to this please read the [Contribution guidelines](CONTRIBUTING.md)

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
