""" Read root file using uproot """
# import time
# from pprint import pprint
import os

import fastjet._pyjet
import numpy as np
import uproot


def root_file():
    """define path to root file"""
    workdir = os.path.realpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    )
    file_root = uproot.open(f"{workdir}/data/uproot_pythia.root")
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
        # pt_min = 300
        # eta_max = 0.9
        jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, R)
        cs = fastjet._pyjet.AwkwardClusterSequence(particle_array, jet_def)

        raw_jets = cs.inclusive_jets()  # .to_list()
        # constituent_jets = cs.constituents()#.to_list()
        # jet_selector = fastjet.SelectorPtMin(pt_min) & fastjet.SelectorAbsEtaMax(eta_max)

        # jets_array = np.array([fastjet.PseudoJet(j.px(), j.py(), j.pz(), j.E()) for j in jets])
        # jets_vector = fastjet.vectorPJ()
        # for jet in jets_array:
        #     jets_vector.append(jet)
        #     jets_vector.push_back(jet)

        # selected_jets = jet_selector(jets_vector)

        pt_list = []
        enjet_list = []

        jets = []
        for jet in raw_jets:
            px = jet["px"]
            py = jet["py"]
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

        sorted_jets = sorted(jets, key=lambda jet: -jet["pt"])
        sorted_pt = [jet.pt for jet in sorted_jets]

        return sorted_pt

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


if __name__ == "__main__":
    read_root()
