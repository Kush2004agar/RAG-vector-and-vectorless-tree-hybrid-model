import glob
import pandas as pd
from pathlib import Path
from config import INPUT_DIR
from rag_v3.serving.pipeline import RagV3Pipeline

pipeline = RagV3Pipeline()

def run_pipeline():
    # Find any Excel sheet in the input directory dynamically via glob
    excel_files = glob.glob(str(INPUT_DIR / "*.xlsx"))
    
    if not excel_files:
        print(f"No Excel files found in {INPUT_DIR}.")
        return
        
    # Pick the first one (or modify to process all)
    file_path = excel_files[0]
    print(f"Processing Questions from: {file_path}")
    
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Failed to read Excel file: {e}")
        return
        
    # Assume the sheet has a 'Question' column
    if 'Question' not in df.columns:
        print("Error: The Excel file must contain a column named 'Question'.")
        # For testing, grab first column
        question_col = df.columns[0]
        print(f"Falling back to first column: '{question_col}'")
    else:
        question_col = 'Question'
        
    answers = []
    
    # Go through the first 5 questions for rapid testing
    for idx, row in df.head(5).iterrows():
        q = str(row[question_col])
        if q == 'nan' or not q.strip():
             answers.append("N/A")
             continue
             
        ans = pipeline.answer(q)
        answers.append(ans)
        
    # Save results
    result_df = df.head(5).copy()
    result_df["Generated_Answer"] = answers
    
    output_path = INPUT_DIR / f"answered_{Path(file_path).name}"
    result_df.to_excel(output_path, index=False)
    print(f"\nCompleted! Saved answers to {output_path}")

if __name__ == "__main__":
    run_pipeline()
