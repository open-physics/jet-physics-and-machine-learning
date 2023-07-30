""" Reconstruct and tag jets """
import logging
import os

import fastjet._pyjet
import numpy as np
import uproot

from generate_pp_events_with_pythia import charm_hadrons

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


def find_charm_hadron_in_jets(
    charm_phi, charm_eta, jet_phi_list, jet_eta_list, R, kwargs
) -> int | None:
    charm_jet = None
    delta_phi_list = charm_phi - jet_phi_list
    delta_eta_list = charm_eta - jet_eta_list
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
        charm_jet = np.min(closest_jets)
    elif len(closest_jets) == 1:
        charm_jet = closest_jets[0]
    else:
        return None
    index = np.where(delta_R_list == charm_jet)[0][0]
    return index


def find_indices(pid, values):
    """Finds the indices of the given values in the list `pid`.

    Args:
      pid: The list to search.
      values: The values to find the indices of.

    Returns:
      A list of tuples, where each tuple contains the indices of the corresponding
      value in the list `pid`.
    """

    indices = []
    for i in range(len(pid)):
        if np.abs(pid[i]) in values:
            indices.append((i,))
    return indices


def read_root():
    file_root = root_file()

    f = uproot.recreate(file_name)

    """
    Information related to root file
    # Number of events are stored in a root file as different trees
    # Each event-tree has equal number of branches storing event-information (e.g., pt, eta, phi, pid, mass etc)
    # Different Trees mean different events and corresponding branches store event-information
    """

    events = file_root.keys()
    particle_array = []
    # Loop on no. of events
    for e_index, event in enumerate(events):
        # Assign each event as separate tree
        tree = file_root[event]
        eta = tree["eta"].array()
        phi = tree["phi"].array()
        pid = tree["pid"].array()

        # Make a list of the eta's and phi's of all charm hadrons
        indices = find_indices(pid=pid, values=charm_hadrons.values())
        charm_eta = []
        charm_phi = []
        for index in indices:
            charm_eta.append(eta[index])
            charm_phi.append(phi[index])

        particle_array = tree.arrays()

        # Define jet radius
        R = 0.7
        # Define jet by providing the name of algorithm and R value
        jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, R)
        # Make the clusters by feeding the paricle array as well as "jet def"
        cs = fastjet._pyjet.AwkwardClusterSequence(particle_array, jet_def)
        raw_jets = cs.inclusive_jets()  # .to_list()
        # constituent_jets = cs.constituents()#.to_list()

        pt_list = []
        phi_list = []
        eta_list = []
        jets = []
        charm_list = []
        # loop on all created jets
        for jet in raw_jets:
            px = jet["px"]
            py = jet["py"]
            pz = jet["pz"]
            en = jet["E"]
            pt = np.sqrt(px**2 + py**2)
            eta, phi = calculate_jet_eta_phi(px, py, pz)
            mass_sq = en**2 - (px**2 + py**2 + pz**2)

            jet["pt"] = pt
            jet["eta"] = eta
            jet["phi"] = phi
            jet["mass_sq"] = mass_sq
            # Label each jet with 0, i.e. it does not contain charm meson by default.
            jet["charm"] = 0

            jets.append(jet)
            pt_list.append(pt)
            eta_list.append(eta)
            phi_list.append(phi)
            charm_list.append(0)

        kwargs = {"e_index": e_index, "pid": pid}

        # Find jet index containing the charm meson
        for i in range(len(charm_phi)):
            index = find_charm_hadron_in_jets(
                np.array(charm_phi[i]),
                np.array(charm_eta[i]),
                phi_list,
                eta_list,
                R,
                kwargs,
            )
            # Label jet as 1 for index found from "find_charm_hadron_in_jets" function
            # Now change the label if the jet contains charm hadron
            if index is not None:
                jets[index]["charm"] = 1

        sorted_jets = sorted(jets, key=lambda jet: -jet["pt"])
        f["event_"] = {"jets": sorted_jets}


if __name__ == "__main__":
    read_root()
