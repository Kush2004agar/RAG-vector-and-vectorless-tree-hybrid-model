import argparse
import glob
import pandas as pd
from pathlib import Path
from config import INPUT_DIR, MAX_QUESTIONS, SAVE_PROGRESS_EVERY_N, DEFAULT_QUESTIONS_EXCEL
from chunk_tree_retriever import answer_question


def _pick_default_input_excel() -> Path | None:
    if DEFAULT_QUESTIONS_EXCEL:
        p = Path(DEFAULT_QUESTIONS_EXCEL)
        if p.is_absolute():
            return p if p.exists() else None
        if (INPUT_DIR / p.name).exists():
            return INPUT_DIR / p.name
        if (INPUT_DIR / DEFAULT_QUESTIONS_EXCEL).exists():
            return INPUT_DIR / DEFAULT_QUESTIONS_EXCEL
        return None

    all_xlsx = sorted(glob.glob(str(INPUT_DIR / "*.xlsx")))
    # Never use a previous output as input
    candidates = [Path(f) for f in all_xlsx if not Path(f).name.startswith("answered_")]
    return candidates[0] if candidates else None


def _resolve_input_excel(input_excel: str | None) -> Path | None:
    # 1) CLI override 2) config DEFAULT_QUESTIONS_EXCEL 3) first non-answered .xlsx in INPUT_DIR
    if input_excel:
        p = Path(input_excel)
        if p.is_absolute() and p.exists():
            return p
        if (INPUT_DIR / p.name).exists():
            return INPUT_DIR / p.name
        if (INPUT_DIR / input_excel).exists():
            return INPUT_DIR / input_excel
        return None

    return _pick_default_input_excel()


def run_pipeline(max_questions=None, save_every=None, input_excel=None):
    max_questions = max_questions if max_questions is not None else MAX_QUESTIONS
    save_every = save_every if save_every is not None else SAVE_PROGRESS_EVERY_N

    file_path = _resolve_input_excel(input_excel)
    if not file_path:
        msg = f"No question Excel files found in {INPUT_DIR} (ignoring answered_*.xlsx)."
        if input_excel:
            msg = f"Input file not found: {input_excel} (expected in data/input or as a full path)."
        print(msg)
        return
    print(f"Processing questions from: {file_path}")

    try:
        df = pd.read_excel(str(file_path))
    except Exception as e:
        print(f"Failed to read Excel file: {e}")
        return

    if "Question" not in df.columns:
        question_col = df.columns[0]
        print(f"No 'Question' column; using first column: '{question_col}'")
    else:
        question_col = "Question"

    # Process all rows or up to max_questions
    to_process = df if max_questions is None else df.head(max_questions)
    n_total = len(to_process)
    print(f"Total questions to process: {n_total} (max_questions={max_questions or 'all'})")

    answers = [None] * n_total  # keep same length as dataframe slice
    out_name = f"answered_{file_path.name}"
    output_path = INPUT_DIR / out_name

    for i, (idx, row) in enumerate(to_process.iterrows()):
        q = str(row[question_col])
        if q == "nan" or not q.strip():
            answers[i] = "N/A"
            print(f"[{i + 1}/{n_total}] Skipped (empty question).")
            continue

        print(f"\n[{i + 1}/{n_total}] Processing...")
        try:
            ans = answer_question(q)
            answers[i] = ans
            print("=" * 60)
            print(f"FINAL ANSWER [{i + 1}]:\n{ans}")
            print("=" * 60)
        except Exception as e:
            answers[i] = f"Error: {e}"
            print(f"Error on question {i + 1}: {e}")

        # Periodic save so 500 questions don't get lost on crash
        if save_every and (i + 1) % save_every == 0:
            result_df = to_process.iloc[: i + 1].copy()
            result_df["Generated_Answer"] = answers[: i + 1]
            result_df.to_excel(output_path, index=False)
            print(f"\n--- Progress saved ({i + 1}/{n_total}) to {output_path} ---\n")

    result_df = to_process.copy()
    result_df["Generated_Answer"] = answers
    result_df.to_excel(output_path, index=False)
    print(f"\nCompleted! Saved {n_total} answers to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QA pipeline from Excel questions.")
    parser.add_argument(
        "-n", "--max-questions",
        type=int,
        default=None,
        help="Max number of questions to process (default: from config, e.g. 500; use 0 for all)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Save partial Excel every N questions (default: from config, e.g. 50)",
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default=None,
        help="Input Excel file (filename in data/input or full path), e.g. 'Network Security and 5G Questions.xlsx'",
    )
    args = parser.parse_args()
    max_n = args.max_questions if args.max_questions != 0 else None
    run_pipeline(max_questions=max_n, save_every=args.save_every, input_excel=args.input)
