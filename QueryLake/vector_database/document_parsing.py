# from langchain.docstore.document import Document
from .text_chunking.document_class import Document
from typing import List, Tuple, Union
# from langchain.document_loaders import PyPDFium2Loader
# from langchain.document_loaders.blob_loaders import Blob
# import PyPDF2
from pypdf import PdfReader
from io import BytesIO
from urllib import request
from markdownify import markdownify as md
import re
# pdf_loader = PyPDFium2Loader("/home/user/python_projects/3035/QueryLakeBackend/user_db/pdf_ium_tmp")



# PyPDF2.PdfFileReader()

def parse_PDFs(bytes_in : Union[bytes, BytesIO], 
               return_all_text_as_string : bool = False) -> List[str]:
    if type(bytes_in) is bytes:
        bytes_in = BytesIO(bytes_in)
    reader = PdfReader(bytes_in)
    all_text : List[Tuple[str, dict]] = []
    for i, page in enumerate(reader.pages):
        parts = []

        def visitor_body(text, cm, tm, fontDict, fontSize):
            zoom_ratio = 266*list(page.mediabox)[3]*9/(list(page.mediabox)[2]*16)
            url = "#page=%d&zoom=%d,%.1f,%.1f" % (i+1, zoom_ratio, 0, min(list(page.mediabox)[3], tm[5]+(fontSize*len(text.split("\n")))+5)) # IDK
            url_2 = "#page=%d&zoom=%d,%.1f,%.1f" % (i+1, zoom_ratio, 0, max(0, float(list(page.mediabox)[3])-list(tm)[5]-5))
            parts.append(text)
            all_text.append((text, {
                "location_link_firefox": url,
                "location_link_chrome": url_2,
                "page": i+1
            }))
        page.extract_text(visitor_text=visitor_body)
    if return_all_text_as_string:
        return " ".join([pair[0] for pair in all_text])
    return all_text

async def parse_url(url_in : str) -> Document:
    try:
        resource = request.urlopen(request.Request(url=url_in, headers={'User-Agent': 'Mozilla/5.0'}))
        content =  resource.read().decode(resource.headers.get_content_charset())
        webpage = content
        find_script = webpage.find("<script>")
        while find_script != -1:
            find_end = find_script+webpage[find_script:].find("</script>")
            webpage = webpage[:find_script]+webpage[find_end+len("</script>"):]
            find_script = webpage.find("<script>")
        return re.sub(r"[\n]+", "\n", md(webpage))
    except:
        return None
