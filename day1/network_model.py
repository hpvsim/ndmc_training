'''
Simple agent-based network model in Python

NOTE: How to view the animation depends on what IDE you're using.
Specifically:
- It will work without change from the command line, Spyder, or VS Code.
- For PyCharm, ensure you disable "SciView" before running.
- For Jupyter, set save_movie = True and view the saved movie
  (you might need to run "pip install ffmpeg-python" in a terminal first)
'''

import numpy as np
import sciris as sc
import pylab as pl

sc.options(dpi=200)

# Set parameters
beta = 3 # Infection rate
gamma = 0.5 # Recovery rate
n_contacts = 10 # Number of people each person is connected to
distance = 0.1 # The distance over which people form contacts
I0 = 1 # Number of people initially infected
N = 100 # Total population size
maxtime = 10 # How long to simulate for
npts = 100 # Number of time points during the simulation
seed = 4 # Random seed to use
dt = maxtime/npts # Timestep length
colors = sc.dictobj(S='darkgreen', I='gold', R='skyblue')
save_movie = False # Whether to save the movie (slow)

# Create the arrays -- one entry per timestep
T = np.arange(npts)
S = np.zeros(npts)
I = np.zeros(npts)
R = np.zeros(npts)
time = T*dt
S[0] = N - I0 # Set initial conditions
I[0] = I0
np.random.seed(seed)


# Define each person
class Person(sc.dictobj):

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
class Sim(sc.dictobj):

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
        for t in T[:-1]:
            self.check_infections() # Check which infectious occur
            self.check_recoveries() # Check which recoveries occur
            self.count(t) # Store results

    def plot(self):
        pl.figure()
        pl.plot(time, S, label='Susceptible', c=colors.S)
        pl.plot(time, I, label='Infectious', c=colors.I)
        pl.plot(time, R, label='Recovered', c=colors.R)
        pl.legend()
        pl.xlabel('Time')
        pl.ylabel('Number of people')
        pl.ylim(bottom=0)
        pl.xlim(left=0)
        pl.show()
        
    def animate(self, pause=0.01, save=False):
        anim = sc.animation()
        fig,ax = pl.subplots()
        x,y = self.get_xy()
        for p in self.contacts:
            p0 = p[0]
            p1 = p[1]
            pl.plot([x[p0], x[p1]], [y[p0], y[p1]], lw=0.5, alpha=0.1, c='k')
            
        handles = []
        for t in T[:-1]:
            if pl.fignum_exists(fig.number):
                for h in handles:
                    h.remove()
                handles = []
                counts = sc.dictobj()
                inds = sc.dictobj()
                for key in ['S', 'I', 'R']:
                    inds[key] = self[key][t]
                    counts[key] = len(inds[key])
                    this_x = x[inds[key]]
                    this_y = y[inds[key]]
                    h = ax.scatter(this_x, this_y, c=colors[key])
                    handles.append(h)
                title = f't={t}, S={counts.S}, I={counts.I}, R={counts.R}'
                pl.title(title)
                pl.pause(pause)
                if save:
                    anim.addframe()
        
        if save:
            anim.save(f'network_{distance}.mp4')
        
        
if __name__ == '__main__':
    
    # Create and run the simulation
    sim = Sim()
    sim.run()
    sim.plot()
    sim.animate(save=save_movie)