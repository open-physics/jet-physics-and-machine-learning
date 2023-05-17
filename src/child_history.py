import numpy as np
from chromo.models import Pythia8
from particle import Particle
from visualize_event import pythia_config, run_conditions

evt_kin = run_conditions()
config = pythia_config("pythia_charm")
pythia = Pythia8(evt_kin, seed=1, config=config)

num_events = 10

for event in pythia(num_events):
    charm_chain = []  # start with the parent index
    pid_chain = []
    for i in range(len(event)):
        # find the original outgoing parton
        if event.status[i] == 23 and abs(event.pid[i]) == 4:  # Charm quark PDG code
            charm_chain.append(i)
        if len(charm_chain) == 0:
            continue
        if event.children[i] is None:
            continue
        child_index_list = list(event.children[i] - 1)
        if event.children[i][0] == event.children[i][1]:
            child_index_list.remove(child_index_list[-1])
        charm_chain += child_index_list

    for j in range(len(charm_chain)):
        pid_chain.append(event.pid[charm_chain[j]])
    if len(charm_chain) != 0:
        print(
            f"The event # is {event.nevent}. "
            f"Daughter history of charm quark (particle ID: 4):\n{np.array(pid_chain)}"
        )
    if 421 in pid_chain:
        print(f"There is {Particle.from_pdgid(421)} at {pid_chain.index(421)}")
    if -421 in pid_chain:
        print(f"There is {Particle.from_pdgid(-421)} at {pid_chain.index(-421)}")
