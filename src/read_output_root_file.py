""" Convert root file to csv using uproot """
# import time
from pprint import pprint
import uproot
# import pandas as pd


def root_file():
    """define the path to root file"""
    # file1_root = uproot.open("~/PycharmProjects/jet-physics-and-machine-learning/output_10k.root")
    file_root = uproot.open("~/PycharmProjects/jet-physics-and-machine-learning/uproot_pythia.root")
    return file_root 

    # breakpoint()
   

def read_root():
    file_root = root_file()

    pprint(file_root.keys())
    pprint(file_root.values())

    # some CHECKS on root file:
    # file_root.classnames()
    # file_root["event_0"].show()
    # file_root["event_0"].typenames()
    # file_root["event_0"].all_members
    # file_root["event_0"]["tr_eta"].array()
    # file_root["event_0"]["tr_eta"].array(library="np")
    # file_root["event_0"]["tr_eta"].array(library="pd")


    # no. of events are stored in root file as different trees
    # Each event-tree has no. of branches storing event-info (e.g., pt, eta, phi, pid, mass etc)
    # trees means different events 
    # branches means event-info

    events = file_root.keys()
    # loop on no of events/trees 
    for event in events:
        # assign each event as separate tree
        tree = file_root[event]
        # branches = len(tree)
        # print(f"Number of branches in tree '{event}': {len(tree)}")
        
        # assign branche names 
        branch_names = tree.keys()
        # loop on branches of each tree/event
        for branch_name in branch_names:
            branch = tree[branch_name]
            data = branch.array()

            # print(f"Contents of branch '{branch_name}' of tree '{event}':")
            print(data[:10])

    file_root.close()
    



def root_to_csv():
    # t0 = time.time()
    dfs = []
    tree = file1_root["mytree"]
    for key in tree.keys():
        print(key, tree[key].array())
        # print(key, len(tree[key].array()))
        # print(key)
        value = tree[key].array()
        # data_frame = pd.DataFrame({key: value})
        # dfs.append(data_frame)

        # breakpoint()
    # dataframe = pd.concat(dfs, axis=1)
    # t1 = time.time()
    # print(t1-t0)

    # dataframe.to_csv("amptsm.csv")




if __name__ == "__main__":
    read_root()

