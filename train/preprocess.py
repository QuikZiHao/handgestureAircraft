import argparse
import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", type=str, default=r'model\dataset')
    parser.add_argument("--datafolder", type=str, default=r'model\dataset')

    args = parser.parse_args()

    return args

def check_dir(dir:str) -> bool:
    return os.path.exists(dir)

def main():
    args = get_args()
    dir = args.dir.split('/')

    try:
        file_list = os.listdir(args.datafolder)
    except:
        print("dataset folder no found, please check again")
        exit()
     
    dataset_pathways = [os.path.join(args.datafolder, file) for file in file_list if file.endswith('.csv') and os.path.isfile(os.path.join(args.datafolder, file))]
    #get the data
    data =[]
    for file in dataset_pathways:
        temp = pd.read_csv(file,header=None)
        data.append(temp)
    data = pd.concat(data, ignore_index=True)
    data.sort_values(by=data.columns[0], inplace=True)

    type_func = data.iloc[:, 0].unique().tolist()
    temp = Path(dir[0])
    for path in dir[1:]:
        if check_dir(temp) == False:
            os.makedirs(temp)
        temp = os.path.join(temp,path)

    # Initialize tqdm with total number of iterations
    progress_bar = tqdm(total=len(type_func), desc='Processing')

    for idx in type_func:
        temp = data[data.iloc[:, 0]==idx]
        temp = temp.iloc[:,1:]
        output_file = os.path.join(args.dir, str(idx))
        if check_dir(output_file) == False:
            os.makedirs(output_file)
        output_file = os.path.join(output_file, f'{idx}.csv')
        # Determine if the file already exists
        mode = 'w'  # default to write mode
        if os.path.exists(output_file):
            mode = 'a'  # switch to append mode if file exists
        
        temp.to_csv(output_file, mode=mode, index=False, header=not os.path.exists(output_file))
        progress_bar.update(1)
    
    progress_bar.close()

if  __name__ == "__main__":
    main()

