import numpy as np

def best_agent(agents, task):
    '''
    Returns the index of the agent most suited for the task.
    '''
    scores = agents @ task
    return np.argmax(scores)

def sample_task(N, weighting):
    '''
    Returns a random task capability vector sampled from an
    (ad hoc) probability distribution.
    TODO: Make this better, give more options for how to control the distribution.
    '''

    return weighting * np.random.lognormal(size=(N))

def run_simulation(N, K, r, s, distribution):

    # Agents is an Nx(K+1) array of agent capability vectors.
    # The first (0th) element represents all of humanity, and is
    # initialized with all ones.
    # The other K elements are AI agents, and are initialized with
    # rescaled lognormal values
    # TODO: capabilities can be negative?
    # TODO: consider other ways of initializing the array
    agents = np.random.lognormal(size=(K+1, N)) / 10
    agents[0] = np.ones(N)

    # Let's say there are 1000 tasks.
    tasks = np.array([sample_task(N, distribution) for i in range(1000)])

    for year in range(100):
        time, automated, n_agents = run_year(agents, tasks)
        automated /= 1000
        GDP = 1000 / time
        agents[0] *= (1 + s)
        agents[1:] *= (1 + r)
        print('Year {}: GDP is {}, proportion automated {:.2f}, distinct agents {}'.format(year, GDP, automated, n_agents))

def run_year(agents, tasks):
    total_time = 0
    total_automated = 0
    total_agents = set()

    for t in tasks:
        a = best_agent(agents, t)
        total_time += 1/(agents[a] @ t)
        total_automated += a > 0
        total_agents.add(a)

    return total_time, total_automated, len(total_agents)

def fast_takeoff():
    N = 1 # Only one capability dimension.
    K = 1 # Only one AI agent.
    r = .08 # Gains to AI: 8% per year
    s = .03 # Gains to humans: 3% per year
    weighting = 1 # All capabilities have same weight when sampling tasks
    
    run_simulation(N, K, r, s, weighting)

def hanson():
    N = 100 # Many capability dimensions.
    K = 100 # Many AI agents.
    r = .08 # Gains to AI: 8% per year
    s = .03 # Gains to humans: 3% per year
    weighting = 1

    
    run_simulation(N, K, r, s, weighting)

def shah():
    N = 100 # Many capability dimensions.
    K = 100 # Many AI agents.
    r = .08 # Gains to AI: 8% per year
    s = .03 # Gains to humans: 3% per year
    distribution = 'non-uniform'
    
    run_simulation(N, K, r, s, np.random.lognormal(size=N))
