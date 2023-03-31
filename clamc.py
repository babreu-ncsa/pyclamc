#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 17:58:34 2018
@author: bruno

Updated on Tue Nov 13 10:25:00 2018
- optimized calculation of relative distances (clamc3)
- minimal image convention added (clamc4)

Updated on Mon Nov 20 15:18:00 2018
- radial density function (clamc5) -> not working

Updated on Tue Nov 27 17:26:00 2018
- radial distribution of pairs (clamc6)

Updated on Mon Dec  3 10:21:00 2018
- fixed MC estimator (clamc6)
"""
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import pandas as pd

#%% ### INPUT PARAMETERS ####

dim = 2
nparts = 200
density = 0.50
L = (nparts / density)**(1.0/dim)
#%%########    DISTANCES ###########
    
def full_rel_distance(configuration):
    """
    calculates matrix of relative distances between particles (triangular matrix)
    """    
    reldist = np.zeros((nparts,nparts))
    for i in range(nparts):
        for j in range(0,i):
            dist2 = 0
            for direction in range(dim):
                dist2 += ( minimal_image(configuration[i][direction] - configuration[j][direction]) )**2.0
            reldist[i][j] = dist2**(1.0/dim)
    return reldist

def single_rel_distance(configuration,particle):
    """
    Calculates relative distances between a certain particle and the others.
    """
    reldist = np.zeros(nparts)
    for i in range(nparts):
        dist = 0
        for direction in range(dim):
            dist += ( minimal_image(configuration[particle][direction] - configuration[i][direction]) )**2.0
        reldist[i] = dist**(1.0/dim)
    return reldist

def pbc(pos):
    """
    Periodic boundary conditions, brings particle back to central cell.
    """
    if pos > L/2.0:
        x = pos
        while x > L/2.0:
            x = x - L
        return x
    if pos < -L/2.0:
            x = pos
            while x < -L/2.0:
                x = x + L
            return x
    if -L/2.0 <= pos <= L/2.0:
        return pos
        
    
def check_encroach(newpart,partlist):
    """
    Takes in a list of positions and check distances. Returns boolean for encroachment (hard cores!)
    """
    for part in partlist:
        dist2 = 0
        for direction in range(dim):
            dist2 += (newpart[direction] - part[direction])**2.0
        if dist2**0.5 < 1.0:
            return True
    return False

def minimal_image(x):
    dist = np.absolute(x)
    if dist > L / 2.0:
        return L - dist
    else:
        return dist
#%% #######    ENERGIES #############

def potential(x):
    """
    Potential with two length scales, hard-core and corona.
    """
    hc_scale = 1.0
    sc_scale = 2.5
    pot_scale = 6.25
    pot_contact = 10.0**6.0
    if x <= hc_scale:
       return pot_contact
    if hc_scale < x <= sc_scale:
        return pot_scale
    if x > sc_scale:
        return 0

def pot_energy(configuration):
    """
    Calculates total potential energy from a configuration of particles.
    """
    reldist = full_rel_distance(configuration)
    energy = 0
    for i in range(nparts):
        for j in range(0,i):
            energy += potential(reldist[i][j])
    return energy


def diff_energy(initial,final,particle):
    """
    Calculates difference in energy between two configurations that differ only by a single MC trial movement.
    """
    dist_in = single_rel_distance(initial,particle)
    dist_fi = single_rel_distance(final,particle)
    particles = [i for i in range(len(dist_in)) if i != particle]
    pot_in = 0
    pot_fi = 0
    for i in particles:
        pot_in += potential(dist_in[i])
        pot_fi += potential(dist_fi[i])
        
    return pot_fi - pot_in


#%% ####### MONTE CARLO #########

def mcsweep(configuration,temp):
    """
    Classical Monte Carlo sweep. Returns configuration and flag for accpetation.
    """
    mvpart = int(ss.uniform.rvs(loc=0,scale=nparts))
    delta = 1.0
    trial_config = np.array(configuration)
    for direction in range(dim):
        trial_config[mvpart][direction] = pbc(configuration[mvpart][direction] + ss.uniform.rvs(loc=-delta/2.0,scale=delta)) 
    dE = diff_energy(configuration,trial_config,mvpart)
    boltz = np.exp(-dE / temp)
    if boltz >= 1.0:
        return trial_config, True
    if boltz < 1.0:
        if boltz > ss.uniform.rvs(loc=0,scale=1):
            return trial_config, True
        else:
            return configuration, False
        
        
#%% ######  SIMULATION  #########
            
def set_space():
    """
    sets configuration space for particles uniformly distributed
    """
    tentative = []
    while len(tentative) < nparts:
        trial = []
        for direction in range(dim):
            trial.append(ss.uniform.rvs(loc=-L/2.0,scale=L))
        if not check_encroach(trial,tentative):
            tentative.append(trial)
    initial = pd.DataFrame(tentative)
    initial.to_csv("initial.csv",header=["X","Y"],index=False)        
    return np.array(tentative)

def simulate(configuration,nsweeps,temp):
    """
    Simulation engine, calls mcsweeps several times.
    """
    config = configuration
    acc = 0
    for i in range(nsweeps):
        config, flag = mcsweep(config,temp)
        acc += flag
        print ("Step n. ", i)
    print("Acceptance rate: ", acc/nsweeps)
    final = pd.DataFrame(config)
    filename = "finalT=" + str("%.1f" % temp) + ".csv"
    final.to_csv(filename,header=["X","Y"],index=False)
    return config

def get_data(configuration, nsweeps, nblocks, temp, density_flag=True, energy_flag=True, gr_flag=True):
    """
    after equilibration, gathers properties of interest.
    So far:
        - potential energy
        - radial density and fourier transform
        - pair distribution
    """
    config = configuration
    global r
    global grid
    densities = []
    kdensities = []
    energies = []
    grs = []
    for i in range(nblocks):
        rho = np.zeros(len(r))
        krho = np.zeros(len(r),dtype=np.complex_)
        en = 0
        gr = np.zeros(len(r))
        for j in range(nsweeps):
            config, flag = mcsweep(config,temp)
            if(density_flag):
                x = radial_density(config,grid) 
                rho += x
                krho += reciprocal_density(x)
            if(energy_flag):
                en += pot_energy(config)
            if(gr_flag):
                gr += pair_distribution(config)
        densities.append(np.divide(rho,nsweeps))
        kdensities.append(np.divide(krho,nsweeps))
        energies.append(en/nsweeps)
        grs.append(np.divide(gr,nsweeps))
        print("Block n. %d" % i)
    return config, np.array(densities), np.array(kdensities), np.array(energies), np.array(grs)

#%%  ######### PHYSICAL QUANTITIES ######

def create_density_grid():
    """
    Creates grid to estimate radial density. Returns:
        - bins edges
        - bins centers
        - areas associated to each bin
    """
    dr = np.power(density,(-1.0/dim))
    nbins = int((L/2.0) / dr)
    bins = np.linspace(0,L/2,nbins+1)
    r = np.zeros(nbins)
    area = np.zeros(nbins)
    for i in range(len(bins)-1):
        r[i] = 0.5*(bins[i] + bins[i+1])
        area[i] = 2.0 * np.pi * r[i] * (bins[i+1] - bins[i])
    return bins, r, area

def create_alternate_density_grid():
    """
    Creates a grid such that the area in between the edges has at least one particle, in average.
    """
    global L
    grid = []
    nbins = int(L*L*np.pi*density / 4.0) - 1
    for i in range(nbins + 1):
        grid.append( np.sqrt(float(i)/(np.pi*density)) )
    r = []
    area = []
    for i in range(nbins):
        r.append( 0.5*(grid[i] + grid[i+1]) )
        area.append( np.pi*(np.square(grid[i+1]) - np.square(grid[i])) )
    return np.array(grid), np.array(r), np.array(area)

def create_boring_density_grid():
    """
    Creates a grid with 100 bins. Returns bins edges, bins centers and bins area.
    """
    global L
    grid = np.linspace(0.0,L/2.0,101)
    nbins = 100
    area = np.zeros(100)
    r = np.zeros(100)
    for i in range(nbins):
        r[i] = 0.5*(grid[i] + grid[i+1]) 
        area[i] = ( np.pi*(np.square(grid[i+1]) - np.square(grid[i])) )
    return grid, r, area


def radial_density(configuration,grid):
    """
    NOT WORKING PROPERLY!!
    Calculates radial density of a certain configuration given the density grid.
    Returns number of partices in each bin of the grid.
    """
    global area
    distances = []
    for i in range(nparts):
        d = np.sqrt(np.sum(np.power(configuration[i,:],2.0)))
        if d <= L/2.0:
            distances.append(d)
    distances = np.array(distances)
    counts, bins = np.histogram(distances, bins=grid)
    return np.divide(counts,area)


def reciprocal_density(dens_vector):
    """
    NOT WORKING PROPERLY!!
    Takes in a vector of densities and Fourier-transforms it.
    """
    N = len(dens_vector)
    pii2N = 2.0 * np.pi * 1j / N
    nk = np.zeros(N,dtype=np.complex_)
    for f in range(N):
        n_k = 0
        for n in range(N):
            n_k += dens_vector[n] * np.exp(-f * n * pii2N)
        nk[f] = n_k
    return nk


def pair_distribution(configuration):
    """ 
    Calculates g(r) of a certain configuration given a density grid.
    Returns number of particles in each bin of the grid.
    """
    global area
    global grid
    gr = np.zeros(len(grid) - 1)
    reldist = full_rel_distance(configuration)
    nz = np.nonzero(reldist)
    gr, bins = np.histogram(reldist[nz], bins=grid)
    gr = np.divide(gr, area*density*float(nparts)/2.0)
    return gr


#%% ######## ESTIMATORS ############
def estimator(x):
    """
    takes in numpy array and provides MC estimator and error according to
    http://ib.berkeley.edu/labs/slatkin/eriq/classes/guest_lect/mc_lecture_notes.pdf
    """
    mean_gx = 0.0
    ss_gx = 0.0
    var_of_mean_gx = 0.0
    estimator = []
    error = []
    for i in range(len(x)):
        gx = x[i]
        mean_gx += (gx - mean_gx) / (float(i) + 1.0)
        ss_gx += gx*gx
        if (i > 0):
            var_of_mean_gx = (ss_gx - (float(i)+1.0)*(mean_gx * mean_gx) ) / (float(i)*(float(i)+1.0))
        estimator.append(mean_gx)
        error.append(np.sqrt(var_of_mean_gx))
    return np.array(estimator), np.array(error)

def estimate_density(densities):
    """
    takes in a np 2d array of densities and returns vectors of estimates and errors.
    """
    global r
    npoints = densities.shape[1]
    avs = []
    errs = []
    for i in range(npoints):
        av, err = estimator(densities[:,i])
        avs.append(av[-1])
        errs.append(err[-1])
    plt.title("Radial density")
    plt.ylabel("n(r)")
    plt.xlabel("r")
    plt.errorbar(r,avs,yerr=errs,fmt="bo-")
    plt.show()
    return avs, errs

def estimate_kdensity(kdensities):
    """
    Same as estimate_density but in reciprocal space. Averages over intensity.
    """
    npoints = kdensities.shape[1]
    k = np.zeros(npoints)
    global L
    for j in range(npoints):
        k[j] = ((2.0 * np.pi) / (L/2.0)) * j
    avs = []
    errs = []
    kdensities = np.absolute(kdensities)
    for i in range(npoints):
        av, err = estimator(kdensities[:,i])
        avs.append(av[-1])
        errs.append(err[-1])
    plt.title("Reciprocal density")
    plt.ylabel("n(k)")
    plt.xlabel("k")
    plt.errorbar(k,avs,yerr=errs,fmt="go-")
    plt.show()
    return avs, errs


def estimate_gr(grs):
    """
    takes in a np 2d array of grs and returns vectors of estimates and errors.
    """
    global r
    npoints = grs.shape[1]
    avs = []
    errs = []
    for i in range(npoints):
        av, err = estimator(grs[:,i])
        avs.append(av[-1])
        errs.append(err[-1])
    plt.title("Radial pair distribution")
    plt.ylabel("g(r)")
    plt.xlabel("r")
    plt.errorbar(r,avs,yerr=errs,fmt="ro-")
    plt.show()
    return avs, errs

#%% ##### PLOTS ###############
def plot_config(configuration,filename, **kwargs):
    x = configuration[:,0]
    y = configuration[:,1] 
    s = 0.5
    c='red'
    vmin=None 
    vmax=None
    
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection

    if np.isscalar(c):
        kwargs.setdefault('color', c)
        c = None
        
    patches = [Circle((x_, y_), s_) for x_, y_, s_ in np.broadcast(x, y, s)]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        collection.set_array(np.asarray(c))
        collection.set_clim(vmin, vmax)
        
    plt.figure()
    ax = plt.gca()
    plt.ylim(-L/2,+L/2)
    plt.xlim(-L/2,+L/2)
    plt.xlabel("X",fontsize=20)
    plt.ylabel("Y",fontsize=20)
    #rscdens = 2.5*2.5*density
    #alpha = rscdens*2.5*2.5
    #plt.title(r'$\sigma_1^2\rho = %.3f$, $\alpha = %.1f$' %  (rscdens, alpha), fontsize=20)
    plt.title("Classical", fontsize=20)
    #ax.annotate('SC', xy=(-0.3,0.95),xycoords='axes fraction',fontsize=24)
    plt.tick_params(labelsize=18)
    plt.gca().set_aspect('equal', adjustable='box')
    ax.add_collection(collection)
    ax.autoscale_view()
    plt.savefig(filename,bbox_inches = "tight")
    if c is not None:
        plt.sci(collection)
    return collection


# set initial configuration
initial = set_space()
# equilibrate
eqb_conf = simulate(initial,5000,5.0)
# calculate g(r)
grid, r, area = create_boring_density_grid()
conf, dens, kdens, ens, grs = get_data(eqb_conf, 100, 10, 0.1, density_flag=False, energy_flag=False)
# save it to file
av, err = estimate_gr(grs)
gr = pd.DataFrame(data=av,columns=["g(r)"])
gr["r"] = r
gr["err"] = err
gr.to_csv("gr.csv")