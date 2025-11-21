import streamlit as st
from PyPDF2 import PdfReader
from io import BytesIO

st.set_page_config(page_title="PDF Viewer")
st.title("PDF Upload & View")

pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

if pdf_file:
    reader = PdfReader(pdf_file)
    text_data = ""


    for page in reader.pages:
        extracted = page.extract_text() or ""
        text_data += extracted + ""


    st.subheader("Extracted Text Preview")
    text_data = text_data.replace("\n","").split(". ")
    for line in text_data:
        st.text(line)