from chromo.models import Pythia8
from visualize_event import pythia_config, run_conditions

evt_kin = run_conditions()
config = pythia_config("pythia_charm")
pythia = Pythia8(evt_kin, seed=1, config=config)

num_events = 10
all_event_chain = []
all_event_dict = {}


def get_family_tree(event, particle):
    pass


for event in pythia(num_events):
    charm_chain = []  # start with the parent index
    pid_chain = []
    particle = {}
    for i in range(len(event)):
        # find the original outgoing parton
        if event.status[i] == 23 and abs(event.pid[i]) == 4:  # Charm quark PDG code
            particle["particle"] = {}
            charm_chain.append(i)
            particle["particle"]["pid"] = event.pid[i]
            parent_index = i - 1
            particle["particle"]["index"] = parent_index

            break

        # # Skip if parent charm not found
        # if len(charm_chain) == 0:
        #     continue

        # # Skip if no children
        # if event.children[i] is None:
        #     continue

        # # Get the child indices (subtract 1 from event-record indices)
        # child_index_list = list(event.children[i] - 1)

        # # Filter as per child statuses
        # if event.children[i][0] == event.children[i][1] == 0:
        #     continue
        # if event.children[i][0] == event.children[i][1] > 0:
        #     child_index_list.remove(child_index_list[-1])
        # if event.children[i][0] > 0 and event.children[i][1] == 0:
        #     continue
        # if event.children[i][0] != event.children[i][1] and min(event.children[i]) > 0:
        #     pass

        # # Add children
        # charm_chain += child_index_list
    particle = get_family_tree(event, particle)

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

    # # Keep a record from all events
    # all_event_chain.append(pid_chain)
    # all_event_dict[f"{event.nevent}"] = pid_dict
