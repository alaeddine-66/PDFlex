import pdfplumber

class SimplePDF:
    def __init__(self, file: str):
        self.file = file

    def page(self, i: int):
        with pdfplumber.open(self.file) as pdf:
            page = pdf.pages[i]
            return page.extract_text()

if __name__ == "__main__":
    reader = SimplePDF("test.pdf")
    print(reader.page(0))
