
from io import BytesIO
import pandas as pd

def dataframe_csv(df):
    return df.to_csv(index=False).encode("utf-8")

def dataframe_excel(df, sheet="Report"):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet[:31])
    return buf.getvalue()

def simple_pdf(title, lines):
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]
    for line in lines:
        story.append(Paragraph(str(line), styles["BodyText"]))
        story.append(Spacer(1, 6))
    doc.build(story)
    return buf.getvalue()
