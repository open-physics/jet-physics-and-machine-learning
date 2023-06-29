""" Read root file using uproot """
# import time
# from pprint import pprint
import numpy as np
import math
import uproot
import fastjet._pyjet
import matplotlib.pyplot as plt

# import seaborn as sns


def root_file():
    """define path to root file"""
    file_root = uproot.open(
        "~/PycharmProjects/jet-physics-and-machine-learning/uproot_pythia.root"
    )
    return file_root


def read_root():
    file_root = root_file()

    # some CHECKS on root file:
    # pprint(file_root.keys())
    # pprint(file_root.values())
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
    particle_array = []
    # loop on no. of events/trees
    for event in events:
        # assign each event as separate tree
        tree = file_root[event]
        particle_array = tree.arrays()

        R = 0.7
        pt_min = 300
        eta_max = 0.9
        jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, R)
        cs = fastjet._pyjet.AwkwardClusterSequence(particle_array, jet_def)

        raw_jets = cs.inclusive_jets()#.to_list()
        constituent_jets = cs.constituents()#.to_list()
        jet_selector = fastjet.SelectorPtMin(pt_min) & fastjet.SelectorAbsEtaMax(eta_max)

        # jets_array = np.array([fastjet.PseudoJet(j.px(), j.py(), j.pz(), j.E()) for j in jets])
        # jets_vector = fastjet.vectorPJ()
        # for jet in jets_array:
        #     jets_vector.append(jet)
        #     jets_vector.push_back(jet)

        # selected_jets = jet_selector(jets_vector)

        pt_list = []
        enjet_list = []
        mjet_list = []

        breakpoint()
        
        jets = []
        for jet in raw_jets:
            px = jet["px"]
            py = jet["py"]
            pz = jet["pz"]
            en = jet["E"]
            pt = np.sqrt(px**2 + py**2)
            # momentum = np.sqrt(px**2 + py**2 + pz**2)
            # mass_squared = en**2 - momentum**2
            
            # if mass_squared >= 0:
            #     mass = np.sqrt(mass_squared)
            # 
            #     threshold = 1.865
            #     if mass < threshold:
            #         print("Jet falls into a specific category based on low mass")
            #     else:
            #         print("Jet falls into a specific category based on high mass")
            # else:
            #     print("mass is negative")


            jet["pt"] = pt
            jets.append(jet)
            pt_list.append(pt)
            enjet_list.append(en)
            
            
        breakpoint()
        sorted_jets = sorted(jets, key=lambda jet: -jet["pt"])



    # print("Clustering with", jet_def.description())
    # # ----------------
    # fig, ax = plt.subplots()  # figsize=(7,5))

    # plt.hist(
    #     pt_list,
    #     bins=50,
    # )
    # plt.xlabel("pt")
    # plt.show()
    # plt.close()
    # plt.savefig("pt.png")



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
