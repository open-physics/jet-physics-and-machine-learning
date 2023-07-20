import awkward as ak
import fastjet
import fastjet._pyjet


def main():
    particles = []
    # an event with three particles:   px    py  pz      E
    # particles.append(fastjet.PseudoJet(99.0, 0.1, 0, 100.0))
    # particles.append(fastjet.PseudoJet(50.0, 0.5, 0, 100.0))
    # particles.append(fastjet.PseudoJet(4.0, -0.1, 0, 5.0))
    # particles.append(fastjet.PseudoJet(10.0, -1, 0, 5.0))
    # particles.append(fastjet.PseudoJet(-99.0, 0, 0, 99.0))

    particles.append(fastjet.PseudoJet(1.2, 31.2, 5.4, 2.5))
    particles.append(fastjet.PseudoJet(3.2, 30.2, 5.4, 2.5))
    particles.append(fastjet.PseudoJet(8.2, 11.2, 5.4, 2.5))
    particles.append(fastjet.PseudoJet(32.2, 64.21, 543.34, 24.12))
    particles.append(fastjet.PseudoJet(32.45, 64.21, 543.14, 24.56))

    array = ak.Array(
        [
            {"px": 1.2, "py": 31.2, "pz": 5.4, "E": 2.5},  # "ex": 0.78},
            {"px": 3.2, "py": 30.2, "pz": 5.4, "E": 2.5},  # "ex": 0.78},
            {"px": 8.2, "py": 11.2, "pz": 5.4, "E": 2.5},  # "ex": 0.78},
            {"px": 32.2, "py": 64.21, "pz": 543.34, "E": 24.12},  # "ex": 0.35},
            {"px": 32.45, "py": 63.21, "pz": 543.14, "E": 24.56},  # "ex": 0.0},
        ]
    )

    # choose a jet definition
    R = 0.7
    jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, R)

    # run the clustering, extract the jets
    fastjet._pyjet.AwkwardClusterSequence(array, jet_def)
    fastjet.ClusterSequence(particles, jet_def)
    # jets = sorted(cs.inclusive_jets(), key=lambda jet: -jet.pt())
    # jet_list1 = cs1.inclusive_jets().to_list()
    # jet_list2 = cs2.inclusive_jets().to_list()

    # print out some infos
    # print("Clustering with", jet_def.description())
    # breakpoint()

    # # print the jets
    # print("        pt y phi")
    # for i, jet in enumerate(jets):
    #     print("jet", i, ":", jet.pt(), jet.rap(), jet.phi())
    #     constituents = jet.constituents()
    #     for j, constituent in enumerate(constituents):
    #         print("    constituent", j, "'s pt:", constituent.pt())


if __name__ == "__main__":
    main()
