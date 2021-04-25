import pickle
import os
import pandas as pd
import glob
import json

def dump(obj, save_path):
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # path = dir_path + "/" + path
    print("PATH", save_path)
    if os.path.exists(save_path):
        os.remove(path=save_path)
    output = open(save_path, 'wb')
    pickle.dump(obj, output)#, protocol=pickle.HIGHEST_PROTOCOL
    output.close()
    # print("type", type(data))
    # datai = pickle.dumps(obj)
    # seq = datai
    # length = int(len(datai) / 10)
    # a = [seq[i:i + length] for i in range(0, len(seq), length)]
    # for i, ai in enumerate(a):
    #     output = open(save_path + str(i), 'wb')
    #     pickle.dump(ai, output)
    #     output.close()
def load(save_path):
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # path = dir_path + "/" + path
    if not os.path.exists(save_path):
        return None
    with open(save_path, 'rb') as file:
        return pickle.load(file)

def load_json(file_path):
    file=open(file_path,"r")
    return json.load(file)

def get_claims(excel_file, sheet_name):  # get values by tag column
    xl = pd.ExcelFile(excel_file)
    print("sheet_names", xl.sheet_names)
    df = xl.parse(sheet_name)
    # data_dict_existed=1 if len(list(data_dict.keys()))>0 else 0

    # df.dropna(subset=["Extractitems"], inplace=True)
    # print("all_columns", df.columns.tolist())
    # current_columns = [x for x in df.columns if
    #                    'Unnamed' not in str(x) and 'No' not in str(x) and "Extractitems" not in str(x)]
    # tag_names = [re.sub("\s+", "", x) for x in df["Extractitems"].values]
    # Folder 1	Folder 2	File	Tag	Related value	Value	Title
    all_claims = df["Claim 1"].values
    return all_claims
def classifaction_report_csv(report, classes, report_name, tag="", write_mode="w"):
    out_dict = {
        "precision": report[0].round(2)
        , "recall": report[1].round(2)
        , "f1-score": report[2].round(2)
        , "support": report[3]
    }
    out_df = pd.DataFrame(out_dict, index=classes)

    avg_tot = (out_df.apply(lambda x: round(x.mean(), 2) if x.name != "support" else  round(x.sum(), 2)).to_frame().T)
    avg_tot.index = ["avg/total"]

    out_df = out_df.append(avg_tot)

def get_file_name_in_dir(folder_name,file_type):
    file_names = glob.glob(folder_name + '/*.'+file_type)
    file_names.sort(reverse=True)
    return file_names
def get_file_name_in_dir_regex(folder_name,ending_txt):
    file_names = glob.glob(folder_name + '/*'+ending_txt)
    file_names.sort(reverse=True)
    return file_names

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def dump_json(obj,save_path):
    with open(save_path, 'w') as outfile:
        json.dump(obj, outfile, ensure_ascii=False, indent=2)
