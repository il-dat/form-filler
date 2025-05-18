#!/bin/bash
# Disable CrewAI and LangChain telemetry/tracking
# Add this to your shell profile to disable tracking globally

# CrewAI tracking
export CREWAI_DO_NOT_TRACK=true

# LangChain tracking
export LANGCHAIN_TRACING_V2=false
export LANGCHAIN_TRACKING=false

echo "CrewAI and LangChain telemetry/tracking disabled successfully."
