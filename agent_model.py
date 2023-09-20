'''
Simple agent-based SIR model in Python
'''

import numpy as np
import pylab as pl

# Set parameters
beta = 2.5 # Infection rate
gamma = 1.0 # Recovery rate
contact_rate = 0.5 # Fraction of population each person is connected to
I0 = 5 # Number of people initially infected
N = 100 # Total population size
maxtime = 10 # How long to simulate for
npts = 100 # Number of time points during the simulation
dt = maxtime/npts # Timestep length

# Create the arrays -- one entry per timestep
x = np.arange(npts)
S = np.zeros(npts)
I = np.zeros(npts)
R = np.zeros(npts)
time = x*dt
S[0] = N - I0 # Set initial conditions
I[0] = I0


# Define each person
class Person:

    def __init__(self):
        self.S = True # People start off susceptible
        self.I = False
        self.R = False

    def infect(self):
        self.S = False
        self.I = True

    def recover(self):
        self.I = False
        self.R = True

    def check_infect(self, other):
        if self.S: # A person must be susceptible to be infected
            if other.I: # The other person must be infectious
                if np.random.rand() < beta/N*dt: # Infection is probabilistic
                    self.infect()
        return

    def check_recovery(self):
        if self.I: # A person must be infected to recover
            if np.random.rand() < gamma*dt: # Recovery is also probabilistic
                self.recover()
        return


# Define the population
class Population:

    def __init__(self):
        self.people = [Person() for i in range(N)] # Create all the people
        for person in self.people[0:I0]: # Make the first I0 people infectious
            person.infect() # Set the initial conditions

    def count_S(self): # Count how many people are susceptible
        return sum([person.S for person in self.people])

    def count_I(self):
        return sum([person.I for person in self.people])

    def count_R(self):
        return sum([person.R for person in self.people])

    def check_infections(self): # Check which infectious occur
        for person1 in self.people:
            contacts = np.random.randint(0, N, int(N*contact_rate))
            for contact in contacts:
                person2 = self.people[contact]
                person1.check_infect(person2)

    def check_recoveries(self): # Check which recoveries occur
        for person in self.people:
            person.check_recovery()


# Create the population
pop = Population()

# Run the simulation
for t in x[:-1]:

    pop.check_infections() # Check which infectious occur
    pop.check_recoveries() # Check which recoveries occur

    S[t+1] = pop.count_S() # Count the current number of susceptible people
    I[t+1] = pop.count_I()
    R[t+1] = pop.count_R()


# Plot
pl.plot(time, S, label='Susceptible')
pl.plot(time, I, label='Infectious')
pl.plot(time, R, label='Recovered')
pl.legend()
pl.xlabel('Time')
pl.ylabel('Number of people')
pl.show()