from docx import Document
d=Document(r"D:\TT paper\1024 TP Intracerebral Hemorrhage\outline.docx")
ps=[p.text.strip() for p in d.paragraphs]
n=0
out=[]
for p in ps:
    if p:
        out.append((n,p[:160])); n+=1
for i,t in out[55:]:
    print(i, "|", t)
