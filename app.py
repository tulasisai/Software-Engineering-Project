import streamlit as st # web 
#pdf handling
import PyPDF2
import fitz
import base64

# os module 
import os
import io
OPENAI_API_KEY=None

from wordcloud import WordCloud

# plots and computer vision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from pdf2image import convert_from_bytes
import cv2
from skimage import io
from PIL import Image 
# import aspose.words as aw
# import ironpdf

# from md2pdf.core import md2pdf
import os
import streamlit as st
from PyPDF2 import PdfReader, PdfWriter 
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback



@st.cache_data
def ChatPDF(text,user_question):
    # st.write(text)
    
    #split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)
    # st.write(chunks)
    # creating embeddings


    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets['API_Key'])
    # st.write("Embedding Created")
    # st.write(embeddings)
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    st.write("Knowledge Base created ")
    #show user input

    def ask_question(i=0):
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            # st.write(docs)

            llm = OpenAI(openai_api_key=st.secrets['API_Key'])
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                # print(cb)
            st.write(response)
            
    ask_question()


def pdfGen(pages, lines, images, word, wordcloudpath, summary, pdftable, filepath):
    # Read the content template from a Markdown file
    with open("format.md", "r") as md_file:
        text_template = md_file.read()

    # Format the text with the provided data
    text = text_template.format(
        pages, lines, images, word, wordcloudpath, 
         summary
    )

    # Create a new PDF document
    doc = fitz.open()

    # Add a new page to the document(
    page = doc.new_page()

    # Add the text data to the page
    page.insert_text((100, 100), text, fontsize=12)

    # Save the PDF document
    doc.save(filepath)
        


import openai
st.set_option('deprecation.showPyplotGlobalUse', False)
from streamlit_javascript import st_javascript


def analyze_color_distribution(image_path):
    image = cv2.imread(image_path)
    hist = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])

    return hist

# make tmp dir
os.makedirs("tmp", exist_ok=True)

# Define the main title for the app
st.title("PDF Evaluation [Team Usability]")

# Create a sidebar menu for navigation
menu = st.sidebar.selectbox("Navigation", ["Guide", "Generate Report","Document Viewing","Font Analysis","Translate","ChatPDF💬","Change PDF for Low Vision","Sentiment Analysis and document classification","NLP English level Analysis","PDF to Audio"])

# Page 1: Guide
if menu == "Guide":
    st.write("## Welcome to the PDF Report Generator and Analyzer.")
    st.write(open("description.txt").read())
    st.write("Please navigate to other pages for specific tasks.")

@st.cache_data
def readFile(uploaded_file):
    print("reading file")
    pdf_file = PyPDF2.PdfReader(uploaded_file)
    pagesCount = (len(pdf_file.pages))
    pdf_text=""
    # progress_text = "Reading Pdf Text... In Progress "
    # my_bar = st.progress(0.0, text=progress_text)
    try:
        for page_num,page in  enumerate(pdf_file.pages):
                pdf_text+="\n"
                pdf_text += page.extract_text()
                percent = (page_num )/pagesCount
                print(page_num,page.extract_text())
                if page_num>6:
                    break
                # my_bar.progress(percent,text = progress_text+" "+str(percent*100) )
        # my_bar.empty()
    except:
        pass
    return pdf_text

@st.cache_data
def imagesExtract(uploaded_file):
    count = 0
    img_data = []
    pdf_file = PyPDF2.PdfReader(uploaded_file)
    try:
        for page_num,page in  enumerate(pdf_file.pages):
            for image_file_object in page.images:
                fileName = str(count) + image_file_object.name
                image_file_name = os.path.join(folderName,fileName)
                with open(image_file_name, "wb") as fp:
                    fp.write(image_file_object.data)
                    count += 1
                # st.image(image_file_name)
                img_data.append(fileName)
    except:
        pass
    return img_data

@st.cache_data
def generate_word_cloud(text,filename):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(8, 4))
    st.write("## word cloud")
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.savefig("{}.png".format(filename), format="png")
    plt.show()
    st.image("{}.png".format(filename))
    return "{}.png".format(filename)
    

@st.cache_data
def pdf_to_images(pdf_path):
    doc = fitz.open(pdf_path)
    image_paths = []
    for page_num in range(len(doc)):
        # st.write(page_num)
        page = doc.load_page(page_num)
        image = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
        # image_bytes = image.getBits()
        # image_data = np.frombuffer(image_bytes, dtype=np.uint8)
        # image_data = image_data.reshape(image.height, image.width, 3)
        # image_pil = Image.fromarray(image_data)
        image_path = f"page_{page_num + 1}.png"
        # image_pil.save(image_path)
        # st.write(dir(image))
        image.save(str(image_path))
        image_paths.append(image_path)
    return image_paths

@st.cache_data
def extract_colors_from_pdf(pdf_path):
    monochrome_pages = 0
    unicolor_pages = 0
    doc = fitz.open(pdf_path)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        img = page.get_pixmap()
        
        if img.is_monochrome:
            monochrome_pages += 1
        else:
            unicolor_pages += 1
    
    return monochrome_pages, unicolor_pages


def plot_histogram(monochrome_pages, unicolor_pages):
    labels = ['Monochrome', 'Unicolor']
    page_counts = [monochrome_pages, unicolor_pages]
    
    plt.bar(labels, page_counts)
    plt.xlabel('Page Type')
    plt.ylabel('Page Count')
    # plt.title('Monochrome vs. Unicolor Pages')
    # plt.show()
    return plt

@st.cache_data
def openaisummarize(pdftext,wordcount):
    openai.api_key = st.secrets['API_Key']
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": f"summarize following content for me in {wordcount} words:\n"+pdftext[:4000]}],
    temperature=0,
    max_tokens=1024
    )
    return response["choices"][0]["message"]["content"]



# Page 2: Generate Report
if menu == "Generate Report":
    st.header("Generate Report from PDF")

    # Allow the user to upload a PDF file
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # create sub floder in tmp on uploaded file name without extension
        filename, _ = os.path.splitext(uploaded_file.name)
        folderName = os.path.join("./tmp",filename)
        try:
            os.mkdir(folderName)
        except FileExistsError:
            pass
        st.write("PDF file uploaded successfully!")

        st.subheader("Report from PDF:")
        
        # Extract text from the uploaded PDF
        pdf_text = ""
        pdf_file =  PyPDF2.PdfReader(uploaded_file)

        # st.write(dir(pdf_file))
        pagesCount = (len(pdf_file.pages))
        
        page = None

        pdf_text=readFile(uploaded_file)

        st.write(pdf_file.metadata)
        
        with st.expander("Extracted Text"):
            st.write(pdf_text)

        summary = None
        with st.expander("Summary"):
            # select number of words
            numWords = st.number_input('Number of Words', min_value=10, max_value=200)
            summary =  openaisummarize(pdf_text,30)
            st.write(openaisummarize(pdf_text,numWords))
        
        # images
        img_data = imagesExtract(uploaded_file)
        # st.write(dir(page))


        
        with st.expander("Extracted Images"):
            if len(img_data)>0:
                options = st.multiselect("Select images to dispaly",img_data)
                # if(len(options)>3):
                cols = st.columns(3)
                for count,option in enumerate(options):
                    cols[count%3].image(os.path.join(folderName,option))
            st.write(img_data)

        
        st.write("## Number of pages:",pagesCount)
        lines = pdf_text.split("\n")
        st.write("## line count :",len(lines))
        words = []
        for i in lines:
            words.extend(i.split(" "))
        st.write("## Word count :",len(words))
        st.write("## Images count :",len(img_data))

        cloudpath = ""
        if(len(words)): cloudpath=generate_word_cloud(pdf_text,filename)
        else: st.write("Cannot generate Word cloud, No text found in Pdf")

        # st.write("## word cloud")
        # pdfGen(pages,lines,images,word,wordcloudpath,summary,pdftable,fielpath)
        pdfGen(
            pagesCount,len(lines),len(words),len(img_data),"![]({})".format(cloudpath),summary,"",f"{filename}.pdf"
        )

        with open(f"{filename}.pdf", "rb") as pdf_file:
            PDFbyte = pdf_file.read()

        st.download_button(label="Export_Report",
                            data=PDFbyte,
                            file_name="report.pdf",
                            mime='application/octet-stream')


        # st.image("{}.png".format(filename))
        with open(os.path.join(uploaded_file.name),"wb") as f: 
                f.write(uploaded_file.getbuffer()) 


        st.pyplot(plot_histogram(*extract_colors_from_pdf(uploaded_file.name)))

        # image analysis

        if uploaded_file:
            st.markdown("### Page Selection")
            # st.write(dir(uploaded_file))
            

            selected_page = st.selectbox("Select a page to analyze", list(range(1, len(fitz.open(uploaded_file.name)) + 1)))
            a = st.empty()

            if st.button("Analyze"):
                a.empty()
                a.cols= a.columns(2)
                st.markdown(f"Analyzing page {selected_page}")
                image_paths = pdf_to_images(uploaded_file.name)
                # a.write(image_paths)
                a.cols[0].image(image_paths[selected_page - 1], use_column_width=True,caption=f"Page {i}")
                import cv2
                import matplotlib.pyplot as plt
                image = cv2.imread(image_paths[selected_page - 1])
                for i, col in enumerate(['b', 'g', 'r']):
                    hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                    # 
                    plt.plot(hist, color = col)
                    plt.ylim([0.0,0.4*1e6])
                    plt.xlim([0, 256])
                    
                a.cols[1].pyplot(plt)

        

def displayPDF(upl_file, width):
    # Read file as bytes:
    bytes_data = upl_file.getvalue()
    # Convert to utf-8
    base64_pdf = base64.b64encode(bytes_data).decode("utf-8")
    # Embed PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width={str(width)} height={str(width*4/3)} type="application/pdf"></iframe>'
    # Display file
    st.markdown(pdf_display, unsafe_allow_html=True)


if menu == "Document Viewing":
    st.header("View Document")
    uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])
    if uploaded_file is not None:
        ui_width = st_javascript("window.innerWidth")
        displayPDF(uploaded_file, ui_width -2)


if menu == "Font Analysis":
    st.header("Font Analysis Page")
    uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])

    if uploaded_file is not None:
        filename, _ = os.path.splitext(uploaded_file.name)
        folderName = os.path.join("./tmp",filename)
        try:
            os.mkdir(folderName)
        except FileExistsError:
            pass

        # Save the uploaded PDF file to a temporary location
        with open(os.path.join(folderName, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.write("PDF file uploaded successfully!")

        st.subheader("Font Analysis:")

        pdf_file_path = os.path.join(folderName, uploaded_file.name)

        doc = fitz.open(pdf_file_path)
        font_info = {}

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)  # Load the page
            fonts = page.get_fonts()
            page_str = str(page_num+1)
            st.write("Page  " + page_str + " font analysis")
            data_frame = pd.DataFrame(fonts,columns=["Font Size","Font type","Font Sub Type","Font Name","Font Descriptor","Font encoding type"])
            # print(type(data_frame))
            st.dataframe(data_frame)
        doc.close()




if menu == "Translate":
    st.header("Text Translation Page")
    uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])

    destination_languages = {
        'afrikaans': 'af', 'albanian': 'sq', 'amharic': 'am', 'arabic': 'ar', 'armenian': 'hy', 'assamese': 'as', 'aymara': 'ay', 'azerbaijani': 'az', 'bambara': 'bm', 'basque': 'eu', 'belarusian': 'be', 'bengali': 'bn', 'bhojpuri': 'bho', 'bosnian': 'bs', 'bulgarian': 'bg', 'catalan': 'ca', 'cebuano': 'ceb', 'chichewa': 'ny', 'chinese (simplified)': 'zh-CN', 'chinese (traditional)': 'zh-TW', 'corsican': 'co', 'croatian': 'hr', 'czech': 'cs', 'danish': 'da', 'dhivehi': 'dv', 'dogri': 'doi', 'dutch': 'nl', 'english': 'en', 'esperanto': 'eo', 'estonian': 'et', 'ewe': 'ee', 'filipino': 'tl', 'finnish': 'fi', 'french': 'fr', 'frisian': 'fy', 'galician': 'gl', 'georgian': 'ka', 'german': 'de', 'greek': 'el', 'guarani': 'gn', 'gujarati': 'gu', 'haitian creole': 'ht', 'hausa': 'ha', 'hawaiian': 'haw', 'hebrew': 'iw', 'hindi': 'hi', 'hmong': 'hmn', 'hungarian': 'hu', 'icelandic': 'is', 'igbo': 'ig', 'ilocano': 'ilo', 'indonesian': 'id', 'irish': 'ga', 'italian': 'it', 'japanese': 'ja', 'javanese': 'jw', 'kannada': 'kn', 'kazakh': 'kk', 'khmer': 'km', 'kinyarwanda': 'rw', 'konkani': 'gom', 'korean': 'ko', 'krio': 'kri', 'kurdish (kurmanji)': 'ku', 'kurdish (sorani)': 'ckb', 'kyrgyz': 'ky', 'lao': 'lo', 'latin': 'la', 'latvian': 'lv', 'lingala': 'ln', 'lithuanian': 'lt', 'luganda': 'lg', 'luxembourgish': 'lb', 'macedonian': 'mk', 'maithili': 'mai', 'malagasy': 'mg', 'malay': 'ms', 'malayalam': 'ml', 'maltese': 'mt', 'maori': 'mi', 'marathi': 'mr', 'meiteilon (manipuri)': 'mni-Mtei', 'mizo': 'lus', 'mongolian': 'mn', 'myanmar': 'my', 'nepali': 'ne', 'norwegian': 'no', 'odia (oriya)': 'or', 'oromo': 'om', 'pashto': 'ps', 'persian': 'fa', 'polish': 'pl', 'portuguese': 'pt', 'punjabi': 'pa', 'quechua': 'qu', 'romanian': 'ro', 'russian': 'ru', 'samoan': 'sm', 'sanskrit': 'sa', 'scots gaelic': 'gd', 'sepedi': 'nso', 'serbian': 'sr', 'sesotho': 'st', 'shona': 'sn', 'sindhi': 'sd', 'sinhala': 'si', 'slovak': 'sk', 'slovenian': 'sl', 'somali': 'so', 'spanish': 'es', 'sundanese': 'su', 'swahili': 'sw', 'swedish': 'sv', 'tajik': 'tg', 'tamil': 'ta', 'tatar': 'tt', 'telugu': 'te', 'thai': 'th', 'tigrinya': 'ti', 'tsonga': 'ts', 'turkish': 'tr', 'turkmen': 'tk', 'twi': 'ak', 'ukrainian': 'uk', 'urdu': 'ur', 'uyghur': 'ug', 'uzbek': 'uz', 'vietnamese': 'vi', 'welsh': 'cy', 'xhosa': 'xh', 'yiddish': 'yi', 'yoruba': 'yo', 'zulu': 'zu'
    }

    if uploaded_file is not None:
        filename, _ = os.path.splitext(uploaded_file.name)
        folderName = os.path.join("./tmp",filename)
        try:
            os.makedirs(folderName, exist_ok=True)
        except FileExistsError:
            pass

        # Save the uploaded PDF file to a temporary location
        with open(os.path.join(folderName, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.write("PDF file uploaded successfully!")

        st.subheader("Document Translation:")

        pdf_file = PyPDF2.PdfReader(uploaded_file)

        selected_language = st.selectbox("Select Destination Language", list(destination_languages.keys()))
        dest_language = destination_languages[selected_language]

        for page_num, page in enumerate(pdf_file.pages):
            pdf_text = page.extract_text()
            with st.expander(f"original page {page_num+1}"):
                st.write(pdf_text, dest_language)
            from deep_translator import GoogleTranslator
            translated = GoogleTranslator(source='auto', target=selected_language).translate(pdf_text)

            with st.expander(f"Translated Text for Page {page_num + 1} ({selected_language}):"):
                st.write(translated)

if menu == "ChatPDF💬":

    st.header("Pdf Content Query : AI supported ")
    uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])
    if uploaded_file is not None:
        st.write("PDF file uploaded successfully!")
        st.subheader("Document Query:")
        pdf_file = PyPDF2.PdfReader(uploaded_file)
        pdf_text= ""
        for page_num, page in enumerate(pdf_file.pages):
            pdf_text +="\n"+ page.extract_text()
        
        user_question = st.text_input("Ask a question about your PDF?")
        if user_question:
            ChatPDF(pdf_text,user_question)

import fitz  # PyMuPDF

import random,string

def displayPDF2(upl_file, width):
    # Read file as bytes:
    bytes_data = upl_file.read()
    # Convert to utf-8
    base64_pdf = base64.b64encode(bytes_data).decode("utf-8")
    # Embed PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width={str(width)} height={str(width*4/3)} type="application/pdf"></iframe>'
    # Display file
    st.markdown(pdf_display, unsafe_allow_html=True)

import random
import string

def update_pdf_for_low_vision(uploaded_file, font_name, font_size):
    filename, file_extension = os.path.splitext(uploaded_file.name)
    folder_name = f"./tmp/{filename}"
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))  # Random 6-character string

    # Unique output file name with random suffix
    output_file = f"{folder_name}/{filename}_low_vision_{random_suffix}.pdf"

    # Create a folder to save the modified PDF
    os.makedirs(folder_name, exist_ok=True)

    pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    writer = fitz.open()

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        new_page = writer.new_page(width=page.rect.width, height=page.rect.height)

        # Get font and size, adjust the size for low vision (increase by 2)
        new_page.insert_text((10, 10), page.get_text("text", clip=page.rect),
                             fontname=font_name, fontsize=font_size)

    writer.save(output_file)
    writer.close()
    pdf_document.close()

    return output_file



if menu == "Change PDF for Low Vision":
    st.header("Change PDF for Low Vision")

    uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])

    if uploaded_file is not None:
        filename, _ = os.path.splitext(uploaded_file.name)
        folder_name = f"./tmp/{filename}"
        try:
            os.makedirs(folder_name, exist_ok=True)
        except FileExistsError:
            pass

        # Save the uploaded PDF file to a temporary location
        with open(f"{folder_name}/{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.write("PDF file uploaded successfully!")

        st.subheader("Transforming PDF for Low Vision")

        # Add dropdown for font style
        font_styles = [
            'Helvetica',
            'Helvetica-Oblique',
            'Helvetica-Bold',
            'Helvetica-BoldOblique',
            'Courier',
            'Courier-Oblique',
            'Courier-Bold',
            'Courier-BoldOblique',
            'Times-Roman',
            'Times-Bold',
            'Times-Italic',
            'Times-BoldItalic',
            'Symbol',
            'ZapfDingbats'
        ]

        font_name = st.selectbox("Font Style", font_styles)

        # Add number input for font size
        font_size = st.number_input("Font Size", min_value=8, max_value=72, value=30)

        updated_pdf_path = update_pdf_for_low_vision(uploaded_file, font_name, font_size)

        ui_width = st_javascript("window.innerWidth")

        modified_file = open(updated_pdf_path, "rb") if os.path.exists(updated_pdf_path) else None

        if modified_file:
            st.header("View Modified PDF for Low Vision")
            displayPDF2(modified_file, ui_width - 2)
        
        st.download_button("Download Modified Pdf File", modified_file, "newPdf.pdf")

from expertai.nlapi.cloud.client import ExpertAiClient


import os
os.environ["EAI_USERNAME"]="techworks009@gmail.com"
os.environ["EAI_PASSWORD"]="spM6gMgqHn!gu6e"


def perform_sentiment_analysis( text, language='en'):
    client = ExpertAiClient()


    return client.specific_resource_analysis(
        body={"document": {"text": text}}, 
        params={'language': language, 'resource': 'sentiment'}
    )

def perform_document_classification( text, taxonomy='iptc', language='en'):
    client = ExpertAiClient()

    return client.classification(
        body={"document": {"text": text}}, 
        params={'taxonomy': taxonomy, 'language': language}
    )

def sentiment_document_calss_results(text):
    result1 = perform_sentiment_analysis(text)
    result2 = perform_document_classification(text)
    return result1,result2

def sentiment_pdf_report_by_txt(text, results, filepath="Sentimentreport.pdf"):
    if text and not results:
        results = sentiment_document_calss_results(text)

    # Initialize a new PDF document
    doc = fitz.open()
    page = doc.new_page()

    # Set the initial cursor position for writing
    cursor_y = 100

    # Write the main text
    page.insert_text((50, cursor_y), "Sentiment Analysis and Document Classification Report", fontsize=15, fontname="helv")
    cursor_y += 30

    # Write the extracted text
    page.insert_text((50, cursor_y), "Extracted Text:", fontsize=12, fontname="helv")
    cursor_y += 20
    
    print(cursor_y)
    page.insert_text((50, cursor_y), text[:1500], fontsize=10)
    cursor_y += 400  # Adjust cursor position based on the length of the text

    print(cursor_y)
    # Write the sentiment analysis result
    sentiment_result = "Overall Sentiment: " + str(results[0].sentiment.overall)
    page.insert_text((50, cursor_y), sentiment_result, fontsize=12)
    cursor_y += 20

    print(cursor_y,end="\n====")

    # Write the document classification results
    page.insert_text((50, cursor_y), "Categories:", fontsize=12, fontname="helv")
    cursor_y += 20
    for category in results[1].categories:
        category_text = f"Category ID: {category.id_}, \n Hierarchy: {category.hierarchy}"
        page.insert_text((100, cursor_y), category_text, fontsize=10)
        cursor_y += 35

    # Save the PDF document
    doc.save(filepath)

    return filepath
    

if menu == "Sentiment Analysis and document classification":
    st.header("Sentiment Analysis and document classification")

    # Allow the user to upload a PDF file
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    


    if uploaded_file:
        text=readFile(uploaded_file)

        with st.expander("Text content"):
            st.write(text)

        output1,output2 = sentiment_document_calss_results(text)
        st.write("Output overall sentiment:")

        st.write(output1.sentiment.overall)

        output = perform_document_classification(text)

        st.write("list of categories:")

        for category in output2.categories:
            st.write("category id: "+str(category.id_), category.hierarchy)
        report_filepath= sentiment_pdf_report_by_txt(text, (output1, output2))

        # Provide a download link for the report
        with open(report_filepath, "rb") as file:
            st.download_button("Download Sentiment Analysis Report", file, "sentiment_analysis_report.pdf")


def extract_text_from_pdf(pdf_path, page_number):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number - 1)
    return page.get_text()

def get_df_tokenization_and_lemmatization(tokens,text):
    # Create a list to hold token data
    data = []
    if tokens is None:
        tokens = perform_pos_lemant_token(text).tokens

    for token in tokens:
        if token.end - token.start >= 8:  # Adjust this condition as needed
            token_text = text[token.start:token.end]
            data.append({'Token': token_text, 'Lemma': token.lemma})

    # Create a DataFrame from the token data
    df = pd.DataFrame(data)

    # Display the DataFrame in Streamlit
    return df

def get_df_pos(tokens,text):
    # Create a list to hold token data
    data = []
    if  tokens is None:
        tokens = perform_pos_lemant_token(text).tokens

    for token in tokens:
        if token.end - token.start >= 8:  # Adjust this condition as needed
            token_text = text[token.start:token.end]
            data.append({'Token': token_text, 'PoS': token.pos})

    # Create a DataFrame from the token data
    df = pd.DataFrame(data)

    # Display the DataFrame in Streamlit
    return df

def get_df_entities(entities,text=None):
    if entities is None:
        entities = perform_entities(text).entities
    # Create a list to hold token data
    data = []

    for token in entities:
        data.append({'Entity': token.lemma, 'Type': token.type_})

    # Create a DataFrame from the token data
    df = pd.DataFrame(data)

    # Display the DataFrame in Streamlit
    return df

def get_df_relations(relations,text=None):
    if relations is None:
        relations = perform_relations(text).relations
    # Create a list to hold token data
    data = []

    # for token in entities:
    #     data.append({'Entity': token.lemma, 'Type': token.type_})
    for rel in relations:
        Verb = rel.verb.lemma
        for r in rel.related:
            data.append({"Verb":Verb,"Relation": r.relation, "Lemma": r.lemma })

    # Create a DataFrame from the token data
    df = pd.DataFrame(data)

    # Display the DataFrame in Streamlit
    return df

def perform_pos_lemant_token(text):
    client = ExpertAiClient()
    language= 'en'
    return client.specific_resource_analysis(
            body={"document": {"text": text}}, 
            params={'language': language, 'resource': 'disambiguation'})

def perform_entities(text):
    client = ExpertAiClient()
    language= 'en'
    return client.specific_resource_analysis(
            body={"document": {"text": text}}, 
            params={'language': language, 'resource': 'entities'})


def perform_relations(text):
    client = ExpertAiClient()
    language= 'en'
    return client.specific_resource_analysis(
            body={"document": {"text": text}}, 
            params={'language': language, 'resource': 'relations'})

def get_df_topic_analysis(text):
    data = []
    taxonomy='iptc'
    client = ExpertAiClient()
    language= 'en'
    document = client.classification(body={"document": {"text": text}}, params={'taxonomy': taxonomy,'language': language})

    categories = []
    scores = []

    # st.write (f'{"CATEGORY":{27}} {"IPTC ID":{10}} {"FREQUENCY":{8}}')
    for category in document.categories:
        categories.append(category.label)
        scores.append(category.frequency)
        # st.write (f'{category.label:{27}} {category.id_:{10}}{category.frequency:{8}}')
        data.append({"CATEGORY":category.label,"IPTC ID": category.id_, "FREQUENCY": category.frequency })
    
    # Create a DataFrame from the token data
    df = pd.DataFrame(data)

    # Display the DataFrame in Streamlit
    return df
    
def get_df_info_extraction(text):
    client = ExpertAiClient()
    language= 'en'
    document = client.detection(
		body={"document": {"text": text}}, 
		params={'language': language,'detector':'pii'})
    data = []
    for extraction in document.extractions:
        # st.write ("Template:", extraction.template)
        for field in extraction.fields:
            # st.write ("field: ", field.name," value: " , field.value)
            for position in field.positions :
                data.append({"Template": extraction.template,"field":field.name,"value": field.value, "start": position.start,"end": position.end })
            #st.write ("start: ", position.start, "end: " , position.end)
    # Create a DataFrame from the token data
    df = pd.DataFrame(data)

    # Display the DataFrame in Streamlit
    return df     

def add_new_page(doc, cursor_y):
    page = doc.new_page()
    return page, 100  # Reset cursor_y to 100 for the new page
def check_cursor_position(doc, page, cursor_y, page_height):
    if cursor_y >= page_height - 100:  # Check if near the bottom of the page
        page, cursor_y = add_new_page(doc, cursor_y)
    return cursor_y, page

import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import os

def dataframe_to_pdf(df, output_filename):
    # Create a temporary plot of the DataFrame
    fig, ax = plt.subplots(figsize=(12, 4))  # Adjust the size as needed
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=df.values, colLabels=df.columns, cellLoc = 'center', loc='center')

    # Save the plot to a PDF
    pp = PdfPages(output_filename)
    pp.savefig(fig, bbox_inches='tight')
    pp.close()

    plt.close(fig)

def create_empty_pdf(message, output_filename):
    """Create a PDF with a message, used for empty DataFrames."""
    with PdfPages(output_filename) as pdf:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, message, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.axis('off')
        pdf.savefig(fig)
        plt.close(fig)

def nlp_level_analysis_pdf_report(text,  filepath="NLPAnalysisReport.pdf"):
    # Initialize a new PDF document
    doc = fitz.open()
    page = doc.new_page()

    # Set the initial cursor position for writing
    cursor_y = 100
    page_height = 792  # Assuming letter size page, adjust as needed

    # Write the main text
    page.insert_text((50, cursor_y), "NLP English Level Analysis Report", fontsize=15, fontname="helv")
    cursor_y += 30

    # Write the extracted text
    page.insert_text((50, cursor_y), "Extracted Text:", fontsize=12, fontname="helv")
    cursor_y += 20

    # Writing the extracted text with a limit to avoid overflow
    page.insert_text((50, cursor_y), text[:1500], fontsize=10)  # Adjust the text slice as needed
    cursor_y += 400  # Adjust cursor position based on the length of the text

    df1 = get_df_tokenization_and_lemmatization(None,text)
    df2 = get_df_pos(None,text)
    df3 = get_df_entities(None,text)
    df4 = get_df_relations(None,text=text)
    df5 = get_df_topic_analysis(text)
    df6 = get_df_info_extraction(text)

    # convert dfs to pdfs and merge pdfs, add headings to orginal pdf all order wise headings in firstpage
    # Generate PDFs for each DataFrame and merge them
    df_filenames = []
    for i, df in enumerate([df1, df2, df3, df4, df5, df6]):
        print(df.shape)
        df_filename = f'temp_df_{i}.pdf'
        if df.empty or 0 in df.shape:
            create_empty_pdf("No content available", df_filename)
        else:
            dataframe_to_pdf(df, df_filename)
        df_filenames.append(df_filename)

    # Merge DataFrames PDFs into the main document
    for df_filename in df_filenames:
        df_doc = fitz.open(df_filename)
        doc.insert_pdf(df_doc)
        df_doc.close()
        os.remove(df_filename)  # Clean up temporary files

    # Save the final merged PDF document
    doc.save(filepath)
    return filepath

if menu == "NLP English level Analysis":
    st.header("NLP English level Analysis")

    # Allow the user to upload a PDF file
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])



    if uploaded_file:
        client = ExpertAiClient()
        
        filename, _ = os.path.splitext(uploaded_file.name)
        folderName = os.path.join("./tmp",filename)
        try:
            os.mkdir(folderName)
        except FileExistsError:
            pass

        # Save the uploaded PDF file to a temporary location
        with open(os.path.join(folderName, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        pdf_file_path = os.path.join(folderName, uploaded_file.name)
        doc = fitz.open(pdf_file_path)

        number_of_pages = doc.page_count

        st.write("Number of pages: ", number_of_pages)

        # get pagenumber input
        pageNumber = int(st.number_input('Enter number of pages', min_value=1,max_value=number_of_pages))

        # Get a specific page
        page = doc.load_page(pageNumber-1)

        # Extract text from the page
        text = page.get_text()
        with st.expander("Text content"):
            st.write(text)

        output = perform_pos_lemant_token(text)
    
        # pos_tags_options = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "NUM", "CONJ", "PRT", "INTJ"]  # Add more POS tags as needed
        pos_tags_options = [
            "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
            "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"
        ]
        selected_pos_tags = st.multiselect("Select POS tags to display", pos_tags_options, default=pos_tags_options)


        with st.expander("Tokenization and Lemmatization"):
            df = get_df_tokenization_and_lemmatization(output.tokens,text)
            st.dataframe(df)
        
        with st.expander("POS tagging"):
            df = get_df_pos(output.tokens,text)
            st.dataframe(df)

        output = perform_entities(text) 
        
        with st.expander(("Enitity vs type")):
            df = get_df_entities(output.entities)
            st.dataframe(df)
            
        # language="en"

        
        # document = perform_relations(text)
    
        with st.expander("Relations Analysis"):
            df = get_df_relations(None,text=text)
            st.dataframe(df)

        with st.expander("topic analysis"):
            df = get_df_topic_analysis(text)
            st.dataframe(df)        

        with st.expander("information extraction"):
            df = get_df_info_extraction(text)
            st.dataframe(df)

        report_filepath= nlp_level_analysis_pdf_report(text)
        with open(report_filepath, "rb") as file:
            st.download_button("Download NLP Linguistic Analysis Report", file, "nlp_linguistic_english_report.pdf")
        
        
if menu == "PDF to Audio":
    st.header("PDF to Audio")
    # Allow the user to upload a PDF file
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    import os
    import PyPDF2 
    from gtts import gTTS
    if uploaded_file:
        folderName = f"./tmp/"

        with open(os.path.join(folderName, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        pdf_file_path = os.path.join(folderName, uploaded_file.name)

        # Read PDF
        path = open(pdf_file_path, 'rb')
        doc = PyPDF2.PdfReader(path)
        # st.write(len(doc.pages))

        # Select a page number (adjust as needed)
        page_num = int(st.number_input('Enter number of pages', min_value=1,max_value=len(doc.pages)))
        from_page = doc.pages[page_num - 1]

        # Extract text from the PDF
        text = from_page.extract_text()

        with st.spinner('Processing the text to speech conversion...'):
            tts = gTTS(text, lang='en')
            audio_file = os.path.join(folderName, "output_audio.mp3")
            tts.save(audio_file)

        with open(audio_file, "rb") as file:
            st.download_button("Download Audio File", file, "output_audio.mp3", "audio/mp3")

        audio_file = open(audio_file, "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/mp3', start_time=0)
