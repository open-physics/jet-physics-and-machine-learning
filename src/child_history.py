from chromo.models import Pythia8
from visualize_event import pythia_config, run_conditions

evt_kin = run_conditions()
config = pythia_config("pythia_charm") + pythia_config("pythia_d0_decay")
pythia = Pythia8(evt_kin, seed=1, config=config)

num_events = 10


def child_status(event, parent_index_list):
    # Get the child indices (subtract 1 from event-record indices)
    # if len(parent_index_list) >= 1:
    i = parent_index_list[0]
    child_index_list = list(event.children[i] - 1)
    # Filter as per child statuses
    if event.children[i][0] == event.children[i][1] > 0:
        child_index_list.remove(child_index_list[-1])
    if event.children[i][0] > 0 and event.children[i][1] == 0:
        child_index_list = []
    if event.children[i][0] != event.children[i][1] and min(event.children[i]) > 0:
        pass
    return child_index_list


for event in pythia(num_events):
    charm_chain = []  # start with the parent index
    pid_chain = []
    for i in range(len(event)):
        # find the original outgoing parton
        if event.status[i] == 23 and abs(event.pid[i]) == 4:  # Charm quark PDG code
            charm_chain.append(i)

            j = 0
            child_index_list = [i]
            while j < 45:
                child_index_list = child_status(event, child_index_list)
                charm_chain += child_index_list
                j += 1

    # for j in range(len(charm_chain)):
    #     pid_chain.append(event.pid[charm_chain[j]])
    # if len(charm_chain) != 0:
    #     print(
    #         f"The event # is {event.nevent}. "
    #         f"Daughter history of charm quark (particle ID: 4):\n{np.array(pid_chain)}"
    #     )
    # if 421 in pid_chain:
    #     print(f"There is {Particle.from_pdgid(421)} at {pid_chain.index(421)}")
    # if -421 in pid_chain:
    #     print(f"There is {Particle.from_pdgid(-421)} at {pid_chain.index(-421)}")
