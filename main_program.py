import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer  # Library TF-IDF
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory  # Library Stemming menggunakan Sastrawi
from sklearn.metrics.pairwise import cosine_similarity  # Mengimpor cosine_similarity untuk menghitung kesamaan antar dokumen
from PyPDF2 import PdfReader  # Mengimpor PdfReader untuk membaca file PDF
from docx import Document  # Mengimpor Document untuk membaca file DOCX

# Fungsi membaca dokumen
def read_file(file_path):
    ext = file_path.split('.')[-1].lower()  # Menentukan ekstensi file
    text = ""
    if ext == "pdf":  # Jika file PDF
        reader = PdfReader(file_path)  # Membaca file PDF
        for page in reader.pages:
            text += page.extract_text()  # Mengekstrak teks dari setiap halaman
    elif ext == "docx":  # Jika file DOCX
        doc = Document(file_path)  # Membaca file DOCX
        text = " ".join([para.text for para in doc.paragraphs])  # Menggabungkan semua teks paragraf
    elif ext == "txt":  # Jika file TXT
        with open(file_path, 'r', encoding='utf-8') as f:  # Membaca file TXT
            text = f.read()  # Membaca seluruh isi file
    else:
        raise ValueError("Format file tidak didukung!")  # Menangani format yang tidak didukung
    return text  # Mengembalikan teks yang diekstrak

# Fungsi memuat daftar stopwords dari file CSV
def load_stopwords(file_path="stopwordbahasa.csv"):
    stopwords = set()  # Membuat set kosong untuk stopwords
    with open(file_path, mode='r', encoding='utf-8') as file:  # Membuka file CSV yang berisi stopwords
        stopwords = set(file.read().splitlines())  # Membaca setiap baris dan memasukkannya ke dalam set
    return stopwords  # Mengembalikan set stopwords

# Fungsi preprocessing
def preprocess(text, stopwords):
    # Case folding (mengubah semua huruf menjadi kecil)
    text = text.lower()
    # Tokenizing (memisahkan teks menjadi kata-kata)
    tokens = re.findall(r'\b[a-z]+\b', text)  # Mengambil kata yang terdiri dari huruf saja
    # Filtering (hapus stopwords)
    filtered_tokens = [token for token in tokens if token not in stopwords]  # Menghapus kata yang ada di stopwords
    # Hapus simbol aneh pada kata asli
    filtered_tokens = [re.sub(r'[^a-zA-Z]', '', token) for token in filtered_tokens]  # Menghapus karakter non-huruf
    # Stemming (mengubah kata ke bentuk dasarnya)
    stemmer = StemmerFactory().create_stemmer()  # Membuat objek stemmer untuk bahasa Indonesia
    stemmed_words_dict = {token: stemmer.stem(token) for token in filtered_tokens}  # Menerapkan stemming
    stemmed_tokens = list(stemmed_words_dict.values())  # Mengambil nilai hasil stemming sebagai list
    
    # Menghitung jumlah kemunculan kata asli
    word_counts = {}  # menyimpan jumlah kemunculan kata
    for word in filtered_tokens:  # Iterasi setiap kata yang difilter
        word_counts[word] = word_counts.get(word, 0) + 1  # Menjumlahkan kemunculan kata
    
    return " ".join(stemmed_tokens), filtered_tokens, stemmed_words_dict, word_counts  # Mengembalikan hasil preprocessing

# Fungsi untuk menjalankan analisis
def analyze_documents(files, query):
    stopwords = load_stopwords("stopwordbahasa.csv")  # Memuat stopwords dari file
    
    documents = []  # List untuk menyimpan teks dokumen yang telah diproses
    all_results = []  # List untuk menyimpan hasil analisis
    
    for file in files:  # Iterasi untuk setiap file yang dianalisis
        text = read_file(file)  # Membaca teks dari file
        processed_text, original_words, stemmed_words_dict, word_counts = preprocess(text, stopwords)  # Melakukan preprocessing
        documents.append(processed_text)  # Menambahkan dokumen yang telah diproses ke dalam list
        
        all_results.append({  # Menyimpan hasil analisis dalam dictionary
            'file': file,  # Nama file
            'original_words': original_words,  # Kata-kata asli dalam dokumen
            'stemmed_words_dict': stemmed_words_dict,  # Kata yang telah di-stem
            'word_counts': word_counts,  # Jumlah kemunculan kata dalam dokumen
        })
    
    # Memproses query
    processed_query, _, _, _ = preprocess(query, stopwords)  # Melakukan preprocessing pada query
    vectorizer = TfidfVectorizer()  # Membuat objek TfidfVectorizer untuk vektorisasi
    X = vectorizer.fit_transform(documents + [processed_query])  # Mengonversi dokumen dan query menjadi vektor TF-IDF
    
    # Menghitung cosine similarity antara query dan setiap dokumen
    similarity_scores = cosine_similarity(X[-1], X[:-1])  # Menghitung kesamaan antara query dan dokumen
    
    # Menambahkan skor kesamaan ke hasil analisis
    for i, result in enumerate(all_results):
        result['probability'] = similarity_scores[0, i]  # Menyimpan skor kesamaan untuk setiap dokumen
    
    # Mengurutkan hasil berdasarkan probabilitas kemiripan (dari terbesar ke terkecil)
    all_results.sort(key=lambda x: x['probability'], reverse=True)
    
    return all_results  # Mengembalikan hasil analisis
