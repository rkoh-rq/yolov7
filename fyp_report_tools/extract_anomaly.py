import argparse
import csv
from pathlib import Path
import pandas as pd

def extract_precision_of_classes(filename):
    result = {}
    classes = ["all"]

    # Open the file and read the contents
    with open(filename, 'r') as f:
        contents = f.read()

    # Split the contents into a list of lines
    lines = contents.split('\n')

    # Iterate through each line
    for line in lines:
        # Split the line by spaces
        split_line = [s for s in line.split(' ') if s != ""]
        if len(split_line) == 0:
            continue
        if split_line[0] in classes:
            P = split_line[3]
            R = split_line[4]
            mAP5 = split_line[5]
            mAP595 = split_line[6]
        elif split_line[0] == "Results":
            if "original" not in line:
                eval_name = line.rsplit("_m7", 1)[0].rsplit("/", 1)[1] 
                eval_name = "".join([e for e in eval_name.split("_") if e != "val" and (e[-4:] == "10.0" or e[-3:] != "0.0")])
                model_name = line.rsplit("/images", 1)[0].rsplit("/", 1)[1].replace("_cityscapes", "")
            else:
                eval_name = "original"
                model_name = line.rsplit("/images", 1)[0].rsplit("/", 1)[1].replace("_original", "")
            model_name = model_name.replace("-cityscapes8", "")
            print(eval_name, model_name)
            if model_name not in result:
                result[model_name] = {}
            result[model_name][eval_name] = mAP595

    pd.DataFrame(result).sort_index().sort_index(axis=1).to_csv("output" + filename.split(".")[0] + ".csv")

    

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("file", help="output txt file")
  args = parser.parse_args()

  # Call the function to extract the numbers
  extract_precision_of_classes(args.file)

