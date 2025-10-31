#!/usr/bin/env python3
"""
PhotoRefine - Advanced Glare and Reflection Removal Tool
Main entry point for the application
"""

import sys
import os

# Add photorefine to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from photorefine.gui.main_window import main

if __name__ == "__main__":
    main()
