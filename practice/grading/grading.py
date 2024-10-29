import argparse
import logging
from pathlib import Path

import nbformat
import nbclient
from nbconvert.preprocessors import ExecutePreprocessor


def extract_student_id(path: Path):
    """ for example, a file path is 'Assignment_3/Assignment#3_20241049_attempt_2024-10-03-08-46-22_20241049_session3.ipynb' """
    id_part = path.stem.split("_attempt")[0]
    student_id = id_part.split("_")[-1]
    return student_id


def execute_notebook(notebook_path, kernel_name):
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name=kernel_name)
    try:
        ep.preprocess(nb)
    except nbclient.exceptions.CellExecutionError:
        logging.info("An error occurs while executing cells")
        nb = None
    return nb
    

def extract_outputs(nb):
    outputs = []
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell_outputs = [output for output in cell.outputs if output.output_type == "execute_result" or output.output_type == "stream"]
            outputs.append(cell_outputs)
    return outputs


def compare_outputs(answer_outputs, student_outputs):
    results = []
    for i, (answer, student) in enumerate(zip(answer_outputs, student_outputs)):
        message = ""
        
        if i == 15:
            message += "Quiz->"
        if not answer or not student:
            results.append(message + "blank output")
            continue
        
        if answer[0].output_type == "stream" and student[0].output_type == "stream":
            if answer[0].text.strip() == student[0].text.strip():
                results.append(message + "o")
            else:
                results.append(message + "x")
                logging.info(f"Wrong in {i}-th cell: \nAnswer: \n{answer[0].text.strip()}\nStudent: \n{student[0].text.strip()}")
        
        elif answer[0].output_type == "execute_result" and student[0].output_type == "execute_result":
            if answer[0]["data"]["text/plain"] == student[0]["data"]["text/plain"]:
                results.append(message + "o")
            else:
                results.append(message + "x")
                logging.info(f"\nWrong in {i}-th cell: \nAnswer: \n{answer[0]['data']['text/plain']}\nStudent: \n{student[0]['data']['text/plain']}")
        
        else:
            raise logging.info("Weird case. Check again.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer_path", type=str, default="./AIP2_Lab03_Linear,_Ridge,_Lasso_script.ipynb")
    parser.add_argument("--files_dir", type=str, default="./Assignment_3")
    parser.add_argument("--output_path", type=str, default="./res.log")
    parser.add_argument("--kernel_name", type=str, default="grading")
    parser.add_argument("--resume_from", type=int, default=-1)
    args = parser.parse_args()
    
    # logger configuration
    logging.basicConfig(
        filename=args.output_path,
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w"
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)
    
    # compare answer vs students'
    submitted_files = list(p for p in Path(args.files_dir).glob("*.ipynb") if not p.stem.endswith("_script"))
    submitted_files = sorted(submitted_files, key=extract_student_id)
    resume_idx = -1
    # if args.resume_from != -1:
    #     for i in range(len(submitted_files)):
    #         student_id = int(str(submitted_files[i]).split("_")[2])
    #         if student_id == args.resume_from:
    #             resume_idx = i
    #             break
    
    answer_nb = execute_notebook(args.answer_path, args.kernel_name)
    answer_outputs = extract_outputs(answer_nb)
    logging.info("Answer output is extracted\n")
    
    for i, file in enumerate(submitted_files):
        if args.resume_from != -1 and i < args.resume_from:
            continue
        logging.info(f"Start comparing {file} ({i}-th student):")
        student_nb = execute_notebook(file, args.kernel_name)
        if student_nb is None:
            logging.info(f"Results for {str(file).split('_')[2]}: Error occured\n")
            continue
        student_outputs = extract_outputs(student_nb)
        
        # compare outputs
        comparison_results = compare_outputs(answer_outputs, student_outputs)
        logging.info(f"Results for {str(file).split('_')[2]}: {comparison_results}\n")