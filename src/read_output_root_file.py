""" Read root file using uproot """
# import time
# from pprint import pprint
import uproot
import fastjet._pyjet
import awkward as ak


def root_file():
    """define the path to root file"""
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
    # loop on no. of events/trees
    for event in events:
        # assign each event as separate tree
        tree = file_root[event]
        # create list of particles
        particles = []
        # loop on branches of each tree/event
        # for particle in tree.arrays():
        #     px = particle["tr_px"]
        #     py = particle["tr_py"]
        #     pz = particle["tr_pz"]
        #     en = particle["tr_en"]

            # particles.append(fastjet.PseudoJet(px, py, pz, en))    
            # particles.append(fastjet.PseudoJet(px, py, pz, en))    


        R = 0.99
        jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, R)
        cs = fastjet._pyjet.AwkwardClusterSequence(tree.arrays(), jet_def)
        # jets = sorted(cs.inclusive_jets(), key=lambda jet: -jet.pt())
        jets = cs.inclusive_jets()
        # breakpoint()
        sorted_jets = []
        for jet in jets:
            sorted_jets.append(jet)

        breakpoint()
        # print("Clustering with", jet_def.description())

    # # print the jets
    # print("        pt y phi")
    # for i, jet in enumerate(jets):
    #     print("jet", i, ":", jet.pt(), jet.rap(), jet.phi())
    #     constituents = jet.constituents()
    #     for j, constituent in enumerate(constituents):
    #         print("    constituent", j, "'s pt:", constituent.pt())
















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
