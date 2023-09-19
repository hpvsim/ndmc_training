"""
Simple agent-based model in Python
Here we how agent beliefs evolve over time.
Agents have two possible beliefs: that the world is flat ("F") or round ("R").
Agents can also be open to change (1=open) or closed (0=closed) about their beliefs
""" 

# Import packages we'll need
import random  # For simulating random numbers
from copy import deepcopy  # For copying
import matplotlib.pyplot as plt  # For plotting


# Define parameters
beliefs = ['F', 'R']  # Flat and round believers

def make_person(belief, is_open):
    return [belief, is_open]

# Part A: Write code to make an agent
class Person:
    def __init__(self, belief, is_open):
        self.belief = belief
        self.is_open = is_open

# We have created a "Person" class. We can use this class to make agents.
# Questions: what does this code do?
# a1 = Agent("F", 0)
# a2 = Agent("R", 1)
# a3 = Agent("R", random.randint(0, 1))


# Part B: Write code to make a population
class Population:

    def __init__(self, N):
        self.N = N  # the population size
        self.people = []  # the list of people in the population
        for i in range(N):  # In this loop, we create all the people
            b = random.randint(0, 1)
            o = random.randint(0, 1)
            agent = Person(beliefs[b], o)
            self.people.append(agent)

    def count_flat_earthers(self):  # Count how many people believe in a flat earth
        return sum([person.belief == "F" for person in self.people])

    def prop_flat_earthers(self):  # The proportion of people who believe in a flat earth
        ### TODO
        # fill in the prop_flat_earthers function
        return self.count_flat_earthers()/self.N

    # Function that randomly selects two agents from the population:
    def interact(self):

        # First, find a random pair to interact
        i = random.randint(0, self.N - 1)
        j = random.randint(0, self.N - 1)
        while i == j:
            j = random.randint(0, self.N - 1)

        # Now we have the speaker and the listener
        speaker = self.people[i]
        listener = self.people[j]

        # Update their beliefs, if the beliefs are not the same and the listener is open
        if listener.belief != speaker.belief:
            if listener.is_open:
                listener.belief = deepcopy(speaker.belief)

        return


# Create a function that simulates a community of size N interacting randomly for K times
def simulate(n, k):
    population = Population(n)
    proportion = []
    for i in range(k):
        population.interact()
        proportion.append(population.prop_flat_earthers())
    return population, proportion


# Simulate 500 interctions between 20 agents
new_population, proportion = simulate(20, 500)
print("Final Population:", new_population)

# Make a plot of the changes in proportion of flat-earthers over interactions
plt.plot(proportion)

# and add some details to the plot
plt.title('Changes in the proportion of F over time')
plt.ylabel('Proportion F')
plt.xlabel('Time [No. of interactions]')
plt.ylim(0, 1)
plt.show()