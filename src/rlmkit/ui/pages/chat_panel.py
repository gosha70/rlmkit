# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.
"""
Chat Panel - Main interactive chat interface
"""

import streamlit as st

# Import all the chat functions from app module
import sys
from pathlib import Path
from rlmkit.ui.app import render_chat_page

# Add parent directory to path to import from app
sys.path.insert(0, str(Path(__file__).parent.parent))

render_chat_page()

# Main entry point for chat panel page
if __name__ == "__main__":
    render_chat_page()
else:
    # When loaded as a page, just call the main function
    render_chat_page()
