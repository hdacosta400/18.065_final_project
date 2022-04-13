# run this to see how much data for good/bad posture we have
import random
import pandas as pd

def good_bad_count():
    with open("./training_data.csv") as csv:
        lines = csv.readlines()
    bad_count = 0 
    good_count = 0

    for l in lines:
        lis = l.strip().split(",")
        if lis[5] == '0':
            bad_count += 1
        else:
            good_count += 1

    print(" good:bad -->",  (good_count, bad_count))

def shuffle_rows():
    # p = pd.read_csv('./training_data.csv', header=None)
    # ds = p.sample(frac=1)
    # ds = ds.astype(float)
    # ds.to_csv('./training_data.csv')
    

    fid = open("./training_data.csv", "r")
    li = fid.readlines()
    fid.close()
    # print(li)

    random.shuffle(li)
    # print(li)

    fid = open("./shuffled_training_data.csv", "w")
    fid.writelines(li)
    fid.close()

shuffle_rows(),