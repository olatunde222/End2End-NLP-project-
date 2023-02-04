# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 19:25:09 2023

@author: Windows 10 Pro
"""
import streamlit as st 
import pandas as pd 
import numpy as np 
import pickle 
import altair as alt



def main():
    
    st.title("Emotion Classifier App")
    menu = ["Home","Monitor","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    
    if choice == "Home":
        st.subheader("Home-Emotion in text")
        
        
        
    elif choice == "Monitor":
        st.subheader("Monitor App") 
        
        
        
    else:
        st.subheader("About")
    
    
    
    
    
    




if __name__ == "__main__" :
    main()
    