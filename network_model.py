'''
Simple agent-based network model in Python
'''

import numpy as np
import sciris as sc
import pylab as pl
sc.options(dpi=200)

# Set parameters
beta = 2.5 # Infection rate
gamma = 1.0 # Recovery rate
n_contacts = 10 # Number of people each person is connected to
distance = 1.0 # The distance over which people form contacts
I0 = 10 # Number of people initially infected
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
        self.x = np.random.rand()
        self.y = np.random.rand()

    def infect(self):
        self.S = False
        self.I = True

    def recover(self):
        self.I = False
        self.R = True

    def check_infection(self, other):
        if self.S: # A person must be susceptible to be infected
            if other.I: # The other person must be infectious
                if np.random.rand() < beta/n_contacts*dt: # Infection is probabilistic
                    self.infect()

    def check_recovery(self):
        if self.I: # A person must be infected to recover
            if np.random.rand() < gamma*dt: # Recovery is also probabilistic
                self.recover()


# Define the simulation
class Sim:

    def __init__(self):
        self.people = [Person() for i in range(N)] # Create all the people
        for person in self.people[0:I0]: # Make the first I0 people infectious
            person.infect() # Set the initial conditions
        self.make_network()
        self.S = []
        self.I = []
        self.R = []
        
    def get_xy(self):
        x = np.array([p.x for p in self.people])
        y = np.array([p.y for p in self.people])
        return x,y
        
    def make_network(self):
        x,y = self.get_xy()
        dist = np.zeros((N,N))
        for i in range(N):
            dist[i,:] = 1 + ((x - x[i])**2 + (y - y[i])**2)**0.5/distance
            dist[i,i] = np.inf
            
        rnds = np.random.rand(N,N)
        ratios = dist/rnds
        order = np.argsort(ratios, axis=None)
        inds = order[0:int(N*n_contacts/2)]
        contacts = np.unravel_index(inds, ratios.shape)
        self.contacts = np.vstack(contacts).T

    def count_S(self): # Count how many people are susceptible
        return sum([person.S for person in self.people])

    def count_I(self):
        return sum([person.I for person in self.people])

    def count_R(self):
        return sum([person.R for person in self.people])

    def check_infections(self): # Check which infectious occur
        for p1,p2 in self.contacts:
            person1 = self.people[p1]
            person2 = self.people[p2]
            person1.check_infection(person2)
            person2.check_infection(person1)

    def check_recoveries(self): # Check which recoveries occur
        for person in self.people:
            person.check_recovery()
    
    def count(self, t):
        this_S = []
        this_I = []
        this_R = []
        for i,person in enumerate(self.people):
            if person.S: this_S.append(i)
            if person.I: this_I.append(i)
            if person.R: this_R.append(i)
        S[t+1] = len(this_S) # Count the current number of susceptible people
        I[t+1] = len(this_I)
        R[t+1] = len(this_R)
        self.S.append(this_S)
        self.I.append(this_I)
        self.R.append(this_R)
            
    def run(self):
        for t in x[:-1]:
            self.check_infections() # Check which infectious occur
            self.check_recoveries() # Check which recoveries occur
            self.count(t) # Store results

    def plot(self):
        pl.figure()
        pl.plot(time, S, label='Susceptible')
        pl.plot(time, I, label='Infectious')
        pl.plot(time, R, label='Recovered')
        pl.legend()
        pl.xlabel('Time')
        pl.ylabel('Number of people')
        pl.show()
        
    def animate(self, pause=0.1):
        pl.figure()
        x,y = self.get_xy()
        for p in self.contacts:
            p0 = p[0]
            p1 = p[1]
            pl.plot([x[p0], x[p1]], [y[p0], y[p1]], lw=0.5, alpha=0.1, c='k')
        for t in x[:-1]:
            inds_S = self.S[t]
            inds_I = self.I[t]
            inds_R = self.R[t]
            
            pl.scatter(x, y)
            
            pl.pause(pause)
        
        
if __name__ == '__main__':
    
    # Create and run the simulation
    sim = Sim()
    sim.run()
    sim.plot()
    sim.animate()