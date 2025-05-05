from flask import Flask, render_template, request
import os
from main_program import analyze_documents
import shutil

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Ambil query dari form
    query = request.form['query']
    
    # Ambil file yang diupload
    uploaded_files = request.files.getlist('documents')
    if not uploaded_files:
        return "Tidak ada dokumen yang dipilih"
    
    # Simpan file sementara di server
    directory = "temp_files"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Menyimpan file yang diupload
    file_paths = []
    for file in uploaded_files:
        file_path = os.path.join(directory, file.filename)
        file.save(file_path)
        file_paths.append(file_path)

    try:
        # Jalankan analisis dokumen
        results = analyze_documents(file_paths, query)
        return render_template('result.html', results=results, query=query)
    except Exception as e:
        return f"Terjadi kesalahan: {e}"
    finally:
        # Cleanup files after processing
        for file_path in file_paths:
            os.remove(file_path)  # Remove file after processing
        if os.path.exists(directory):
            shutil.rmtree(directory)  # Remove temp directory

if __name__ == '__main__':
    app.run(debug=True)