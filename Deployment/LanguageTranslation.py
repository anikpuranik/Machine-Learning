# Importing Libraries
from textblob import TextBlob 
import streamlit as st
import pandas as pd

# Preparing Language
languages = pd.read_csv("Language_and_symbols.txt", delimiter='\t')
languages = languages.sort_values(by=["Language Name"])
languages.index = [i for i in range(len(languages))]

# Logic Designing
def translate(message, target_language):
    message = TextBlob(message)
    translated_message = message.translate(to=target_language)

    return translated_message

# UI Designing
def translation():
    st.title("Language Translation")
    message = st.text_area("Enter the text to be translated")
    
    st.text(languages)
    target_language = st.text_input("Enter the target language")
    
    if st.button("translate"):
        try:
            st.text("Tranlated text is")
            translated_message = translate(message, target_language)
        except:
            translated_message = "Language selected was either not in list or same language is selected"
        
        st.success(translated_message)
        
if __name__ == "__main__":
    translation()
