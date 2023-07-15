""" Read root file using uproot """
# import time
import logging
import os

import fastjet._pyjet
import numpy as np
import uproot

workdir = os.path.realpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
file_name = f"{workdir}/data/uproot_jet_tagging.root"


def root_file():
    """define path to root file"""
    workdir = os.path.realpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    )
    file_root = uproot.open(f"{workdir}/data/uproot_pythia.root")
    return file_root


def calculate_jet_eta_phi(px, py, pz):
    p = np.sqrt(px**2 + py**2 + pz**2)
    cos_theta = pz / p
    theta = np.arccos(cos_theta)
    eta = -np.log(np.tan(theta / 2))
    phi = np.arctan2(py, px)
    return eta, phi


def find_D0meson_in_jets(
    d0_phi, d0_eta, jet_phi_list, jet_eta_list, R, kwargs
) -> int | None:
    D0_jet = None
    delta_phi_list = d0_phi - jet_phi_list
    delta_eta_list = d0_eta - jet_eta_list
    delta_R_list = np.sqrt(delta_eta_list**2 + delta_phi_list**2)
    closest_jets = delta_R_list[delta_R_list < R]
    logging.info(closest_jets)

    """
    # array with number of jets
    # now, find the minimum delta_r value in the array and its index
    # label that index as 1 (one can add dictionary for labelling jet)
    # and label rest of the delta_r values as 0
    """

    if len(closest_jets) > 1:
        D0_jet = np.min(closest_jets)
    elif len(closest_jets) == 1:
        D0_jet = closest_jets[0]
    else:
        return None
    index = np.where(delta_R_list == D0_jet)[0][0]
    return index


def read_root():
    file_root = root_file()

    f = uproot.recreate(file_name)

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

    """
    Information related to root file
    # Number of events are stored in a root file as different trees
    # Each event-tree has equal number of branches storing event-information (e.g., pt, eta, phi, pid, mass etc)
    # Different Trees mean different events and corresponding branches store event-information
    """

    events = file_root.keys()
    particle_array = []
    # loop on no. of events
    for e_index, event in enumerate(events):
        # assign each event as separate tree
        tree = file_root[event]
        eta = tree["eta"].array()
        phi = tree["phi"].array()
        pid = tree["pid"].array()

        # eta and phi of D0meson
        index = np.where(pid == 421)
        d0meson_eta = eta[index]
        d0meson_phi = phi[index]

        particle_array = tree.arrays()

        # define jet radius
        R = 0.7
        # Define jet by providing the name of algorithm and R value
        jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, R)
        # make the clusters by feeding the paricle array as well as "jet def"
        cs = fastjet._pyjet.AwkwardClusterSequence(particle_array, jet_def)
        raw_jets = cs.inclusive_jets()  # .to_list()
        # constituent_jets = cs.constituents()#.to_list()

        pt_list = []
        phi_list = []
        eta_list = []
        jets = []
        D0_list = []
        # loop on all created jets
        for jet in raw_jets:
            px = jet["px"]
            py = jet["py"]
            pz = jet["pz"]
            pt = np.sqrt(px**2 + py**2)
            eta, phi = calculate_jet_eta_phi(px, py, pz)

            jet["pt"] = pt
            jet["eta"] = eta
            jet["phi"] = phi
            jet["D0"] = 0

            jets.append(jet)
            pt_list.append(pt)
            eta_list.append(eta)
            phi_list.append(phi)
            D0_list.append(0)

        kwargs = {"e_index": e_index, "pid": pid}

        # Find jet index containing the D0 meson
        for i in range(len(d0meson_phi)):
            index = find_D0meson_in_jets(
                np.array(d0meson_phi[i]),
                np.array(d0meson_eta[i]),
                phi_list,
                eta_list,
                R,
                kwargs,
            )
            # label jet as 1 for index found from "find_D0meson_in_jets" function
            if index is not None:
                jets[index]["D0"] = 1

        f["event_"] = {"jets": jets}

        # pd.set_option('display.max_rows', len(df))
        sorted_jets = sorted(jets, key=lambda jet: -jet["pt"])
        [jet.pt for jet in sorted_jets]
        # return sorted_pt
    # print("Clustering with", jet_def.description())

    # with uproot.open("uproot_jet_tagging.root") as f1:
    # f1.keys()


if __name__ == "__main__":
    read_root()
