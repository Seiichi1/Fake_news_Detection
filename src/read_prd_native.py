import zipfile
import xml.etree.ElementTree as ET
import os

def get_docx_text(filename):
    if not os.path.exists(filename):
        return "File not found"
    
    try:
        with zipfile.ZipFile(filename) as docx:
            content = docx.read('word/document.xml')
            tree = ET.fromstring(content)
            
            # The namespace for docx is usually this
            namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            
            text = []
            for p in tree.findall('.//w:p', namespaces):
                p_text = []
                for t in p.findall('.//w:t', namespaces):
                    if t.text:
                        p_text.append(t.text)
                text.append(''.join(p_text))
            
            return '\n'.join(text)
    except Exception as e:
        return f"Error reading docx directly: {e}"

if __name__ == "__main__":
    print(get_docx_text("Product Requirements Document.docx"))
