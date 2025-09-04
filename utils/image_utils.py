"""
File utility functions for handling image paths and static resources
"""
import os
import base64
from pathlib import Path
import streamlit as st

def get_base_path():
    """Get the base path of the application"""
    return Path(__file__).parent.parent.parent

def get_image_path(image_name):
    """
    Get absolute path to an image in the root directory
    
    Args:
        image_name (str): Name of the image file
        
    Returns:
        Path: Absolute path to the image
    """
    base_path = get_base_path()
    return base_path / image_name

def get_image_base64(image_name):
    """
    Get base64 encoded image for embedding in HTML/CSS
    
    Args:
        image_name (str): Name of the image file
        
    Returns:
        str: Base64 encoded image data
    """
    try:
        image_path = get_image_path(image_name)
        if not image_path.exists():
            print(f"Image not found: {image_path}")
            return None
            
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        print(f"Error encoding image: {str(e)}")
        return None
