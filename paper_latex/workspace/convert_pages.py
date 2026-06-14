import fitz
import os

WORKSPACE = "/ai-inventor/aii_data/runs/run_PoDi6I8fYcAb/4_gen_paper_repo/_4_assemble_paper/paper/workspace"
pdf_path = os.path.join(WORKSPACE, "paper.pdf")
out_dir = os.path.join(WORKSPACE, "page_screenshots")
os.makedirs(out_dir, exist_ok=True)

doc = fitz.open(pdf_path)
print(f"Total pages: {len(doc)}")
for i, page in enumerate(doc):
    mat = fitz.Matrix(150/72, 150/72)  # 150 DPI
    pix = page.get_pixmap(matrix=mat)
    out_path = os.path.join(out_dir, f"page_{i+1:02d}.png")
    pix.save(out_path)
    print(f"Saved page {i+1} -> {out_path}")
print("Done.")
