""" Convert root file to csv using uproot """

# import time
# from pprint import pprint
from pprint import pprint
import uproot
import pandas as pd


def main():
    """read and convert root to csv file"""
    file_root = uproot.open("~/PycharmProjects/jet-physics-and-machine-learning/output_10k.root")
    pprint(file_root.keys())
    pprint(file_root.values())
    file_root.classnames()
    dfs = []
    tree = file_root["mytree"]
    # t0 = time.time()
    for key in tree.keys():
        print(key, tree[key].array())
        # print(key, len(tree[key].array()))
        # print(key)

        value = tree[key].array()
        # data_frame = pd.DataFrame({key: value})
        # dfs.append(data_frame)

        breakpoint()
    # dataframe = pd.concat(dfs, axis=1)
    # t1 = time.time()
    # print(t1-t0)


    # dataframe.to_csv("amptsm.csv")


if __name__ == "__main__":
    main()
