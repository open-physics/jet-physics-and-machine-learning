""" Read root file using uproot """
# import time
import logging
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


def calculate_jet_eta_phi(px, py, pz):
    p = np.sqrt(px**2 + py**2 + pz**2)
    cos_theta = pz / p
    theta = np.arccos(cos_theta)
    eta = -np.log(np.tan(theta / 2))
    phi = np.arctan2(py, px)
    return eta, phi


def find_D0meson_in_jets(d0_phi, d0_eta, jet_phi_list, jet_eta_list, R) -> float | None:
    D0_jet = None
    delta_phi_list = d0_phi - jet_phi_list
    delta_eta_list = d0_eta - jet_eta_list
    delta_R_list = np.sqrt(delta_eta_list**2 + delta_phi_list**2)
    closest_jets = delta_R_list[delta_R_list < R]
    logging.info(closest_jets)
    if len(closest_jets) > 1:
        D0_jet = np.min(closest_jets)
    elif len(closest_jets) == 1:
        D0_jet = closest_jets[0]
    return D0_jet


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

    """
    Information related to root file
    # Number of events are stored in a root file as different trees
    # Each event-tree has equal number of branches storing event-information (e.g., pt, eta, phi, pid, mass etc)
    # Different Trees mean different events and corresponding branches store event-information
    """

    events = file_root.keys()
    particle_array = []
    # loop on no. of events
    for event in events:
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
        # pt_min = 300
        # eta_max = 0.9
        # Define jet by providing the name of algorithm and R value
        jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, R)
        # make the clusters by feeding the paricle array as well as "jet def"
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
        phi_list = []
        eta_list = []
        jets = []
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

            jets.append(jet)
            pt_list.append(pt)
            eta_list.append(eta)
            phi_list.append(phi)

        find_D0meson_in_jets(d0meson_phi, d0meson_eta, phi_list, eta_list, R)

        # pd.set_option('display.max_rows', len(df))
        sorted_jets = sorted(jets, key=lambda jet: -jet["pt"])
        [jet.pt for jet in sorted_jets]
        # return sorted_pt
    # print("Clustering with", jet_def.description())


if __name__ == "__main__":
    read_root()
