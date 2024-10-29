# Introduction

For grading .ipynb file, run this code

```python
python grading.py --answer_path . --files_dir ./files_dir --output_path ./grading_results.log --kernel_name grading --resume_from 17
```

<details>
    <summary><b>Command line arguments</b></summary>
    <div markdown="1">
        <ul>
            <li>answer_path: path for answer ipynb file</li>
            <li>files_dir: directory path for students' ipynb files</li>
            <li>output_path: path for log file</li>
            <li>kernel_name: ipykernel name to execute ipynb file</li>
            <li>resume_from: index of the students' files to resume (-1 means start from beggining)</li>
        </ul>
    </div>
</details> 

When you give students the assignment, note that:
- assignment template **should not contain any results**, because this code decides the student's codes are true if the student **just print answer value**, instead of the result variables
- remind students to fill in the code in the very cell, not the additional cell.

# TODO
There should be some supplementations to consider:
- sometimes some outputs are skipped even though the original code is correct
- how to grade visual results (plot, picture, ...)
- if there are any blank cells
- if there are any error cells (to skip)