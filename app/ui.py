import streamlit as st
import pandas as pd
from PIL import Image
from backend import classify_text

def display_ui():
    st.set_page_config(page_title="GLTR-based AI Text Detector", layout="wide")

    main_left_col, main_right_col = st.columns([0.5, 0.5], gap="large")

    with main_left_col:
        st.title("GLTR-based AI Text Detector")
        st.write("<br>", unsafe_allow_html=True)
        st.markdown("This is a prototype designed for detecting AI-generated texts based on the [GLTR visual tool](https://arxiv.org/pdf/1906.04043).")
                    
        st.write("<br>", unsafe_allow_html=True)
        st.write("<br>", unsafe_allow_html=True)
        st.write("<br>", unsafe_allow_html=True)
        text = st.text_area("Enter your text here:", height=200)
        get_sql_button = st.button("Give me the answer!")
        
    if get_sql_button:
        with main_right_col:
            st.write("Results")
            if text.strip() == "":
                st.write("Sorry, there is no question provided")
            else:
                colored_text, label, count = classify_text(text)
                st.write(f"Predicted label: ")
                if label == "Human":
                    img = Image.open("images/Human-icon.png")
                    original_width, original_height = img.size
                    new_width = original_width // 5
                    new_height = original_height // 5
                    img = img.resize((new_width, new_height))
                    st.image(img)
                else:
                    img = Image.open("images/Generated-icon.png")
                    original_width, original_height = img.size
                    new_width = original_width // 5
                    new_height = original_height // 5
                    img = img.resize((new_width, new_height))
                    st.image(img)
                st.markdown(colored_text, unsafe_allow_html=True)
                st.write("<br>", unsafe_allow_html=True)
                df_counts = pd.DataFrame(list(count.items()), columns=["Color", "Count"])
                st.table(df_counts.set_index("Color"))
    return