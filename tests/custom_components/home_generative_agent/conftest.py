"""Shared test configuration."""

import os

# langchain-openai >= 1.6 probes a throwaway socket when it builds a
# ChatOpenAI client, to drop TCP keepalive options the kernel rejects. The
# library tolerates a blocked socket, but the HA test plugin fails any test
# that tried to open one. The probe never connects, so turning the keepalive
# options off in tests loses nothing.
os.environ.setdefault("LANGCHAIN_OPENAI_TCP_KEEPALIVE", "0")
