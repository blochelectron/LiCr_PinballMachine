#-------------------------------------------#
#--------#     Pinball  Machine    #--------#
#--------#      ...on Python!      #--------#
#-------------------------------------------#
#--------#
#  load SpinDiff_anal."Pallinometri/PinballMachines_SettingsMathFunctions_1.00.gp"
#--------#
import math as m
import numpy as np
import random
import codecs
from Pallynometro_Header2 import *
#-----#
import sys
sys.stdout.reconfigure(encoding='utf-8')
#sys.path.insert(0, 'C:/Users/stefa/Desktop/PhD/Code/python/')
#from Pallynometro_Header import *
#-----#

import time
t_start = time.time()

# use atoms as vector if True, otherwise not (old code)
use_vectors = True

# if seed = None: each run is different (default).
# if seed = integer: each run is the same and use_vectors = True or False should give exactly same result!
#                    timing might be different and memory usage is higher here.              
seed = None         # default
#seed = 0x12345678   # use this only for debugging!

# test output for comparison when seed is not None. -1 if not used.
if seed is None:
    test_jt = -1
    test_ja = -1
else:    
    test_jt = 1
    test_ja = 16

if use_vectors:
    mfp_file    = './sim/mfp_vectors.dat'
    coords_file = './sim/coord_vectors.dat'
else:
    mfp_file    = './sim/mfp_classic.dat'
    coords_file = './sim/coord_classic.dat'     

# if True save mean free path
save_mfp = True

# if True use np.savetxt to save result
# might be even slower than without?
save_text = False

sep             = ' '
coords_hdr      = ['B(mG)','#','t(ms)','x(µm)','y(µm)','z(µm)','vx(µm/ms)','vy(µm/ms)','vz(µm/ms)']
coords_fmt      = '%9.3f'
coords_fmt_all  = sep.join([coords_fmt]*len(coords_hdr))
coords_hdr      = sep.join([(coords_fmt.split('.')[0]+'s') % h for h in coords_hdr])

mfp_hdr         = ['Dt_free(ms)','Dx_free(µm)','RealDx_free(µm)','Real_l_free(µm)','RetardTime(µs)']
mfp_fmt         = '%15.6f'
mfp_fmt_all     = sep.join([mfp_fmt]*len(mfp_hdr))
mfp_hdr         = sep.join([(mfp_fmt.split('.')[0]+'s') % h for h in mfp_hdr])

#----------------------------------------------# [Numerical Simulator]   ((((SIMPLIFIED))))
#------------------------------#
#-----#  [Auto ScanDet]  #-----#  ((((SIMPLIFIED))))
#------------------------------#
#--------# Notes
# Cross 5V with Python

# changed det from set 096
#--------# "Parallelization" settings
SimSet   = '101'
SimInstn = 12       # instance
#--------# Atomic sample numbers: ((Li))
TGaussLi = 0.40e-6     # T_x
TGauzzLi = 0.40e-6      # T_z, T_y
SX0_Li   = 17.0e-6    
SZ0_Li   = 6.5e-6
SY0_Li   = 7.5e-6     # taking into account effect of trap freqs
SV0_Li   = m.sqrt(k_B*TGaussLi/m_Li6)  # sigma v_x
SV0z_Li  = m.sqrt(k_B*TGauzzLi/m_Li6)  # sigma v_z, v_y
ωxLi     = (2*π) * 16.7   # x = axial (along ODT axis)         # NB: the final one w/o cross
ωyLi     = (2*π) * 451.   # y = radial (don't see, along img)  # NB: the final one w/o cross
ωzLi     = (2*π) * 577.   # z = vertical (along gravity)       # NB: the final one w/o cross (irrelevant here)
#--------# Atomic sample numbers: [[Cr]]
Natom_Cr = 130.e3
sx_Cr    = 110.e-6
sz_Cr    = 9.7e-6
sy_Cr    = 11.0e-6
TGaussCr = 0.50e-6  # NB: HERE for y and z
TGauxxCr = 0.50e-6  # NB: for x
SV0_Cr   = m.sqrt(k_B*TGaussCr/m_Cr53)
SV0x_Cr  = m.sqrt(k_B*TGauxxCr/m_Cr53)
MassRat  = m_Cr53/m_Li6 
#m_red   = m_Li6 * MassRat/(1.+MassRat) 
m_red    = (m_Cr53*m_Li6)/(m_Cr53+m_Li6)
m_LiCr   = (m_Li6 + m_Cr53)
ωxCr     = (2*π) * 13.8   # x = axial (along ODT axis)         # NB: the final one w/o cross
ωyCr     = (2*π) *  88.   # y = radial (don't see, along img)  # NB: the final one w/o cross
ωzCr     = (2*π) * 112.   # z = vertical (along gravity)       # NB: the final one w/o cross (irrelevant here)
#--------# Simulation settings
modecoll = 'Cr_RealT' #  'Cr_M∞', 'Cr_RealT'
modebath = 'Cr_Gauss' #  'Cr_Gauss', 'Cr_homog'
N_atoms  = 10000     # NB: not real atom number, but number of simulated atoms/trajectories/histories
DTSTEP   = [100.e-6, 50.e-6, 25.e-6, 10.e-6,  5.e-6,  2.e-6,  1.e-6]      
def IsCloseToRes_dt(dB):
    if (dB >= -1. and dB <= 4.):
        return 5
    elif (dB>= -3. and dB<=10.):
        return 4
    elif (dB>=-10. and dB<=40.):
        return 3
    elif (dB>=-20. and dB<=60.):
        return 2
    else:
        return 1    # 100 is not used in this case

#def IsCloseToRes_dt(dB):
#    return 5

t_final  = 40.e-3   # [40 ms]
#Nbins   = 45
SaveLastPoint = 0   # keep it 0
RetTimePhase  = 0
MeanFieldDima = 0
#--------# 


#--------# Sampling every 0.5ms: functions
t_05 = 0.5e-3
N_05 = int(t_final/t_05)    # without +1 if j05 starts from 0
def timejt(jt):
    return (jt)*dt_step   # NB: removing -1 wrt gnuplot!  arrays in python start from 0

def jCorrespTo_time(time_ms):
    return int(time_ms/(1.e3*dt_step) + 1.e-4) + 0  # NB: no check if time_ms belongs to the simulation! use with caution

def t05_from_j05(j05):
    return 0.5*j05

#--------# Cr bath 'adjustment'   (python only, not implemented in gnuplot)
if (modebath == 'Cr_homog'):     # if you want to test homogeneous Cr bath
    crbath = 0.0
elif (modebath == 'Cr_Gauss'):
    crbath = 1.0

#--------#

#------------------------------------# Create detuning axis + Initialization of some stuff
#-------# Detuning axis (choose)
if (SimInstn == 0): 
    δB_AXIS = [ -100., -90., -80., -70., -60., -50., -40.,       \
                 -35., -30., -25., -20., -18., -16., -14., -12., \
                 -10.,  -9.,  -8.,  -7.,  -6.,  -5.,  -4.,  -3.,  -2.,  -1., \
                   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10., \
                  12.,  14.,  16.,  18.,  20.,  22.,  24.,  26.,  28.,  30.,  35., \
                  40.,  50.,  60.,  70.,  80.,  90.,  100.  ]

#-------#
if (SimInstn == 1):      # (1), 65 comp_t_units
    δB_AXIS = [-100., -95., -90., -85., -80., -75., -70., -65., -60., 
                -55., -50., -45., -40., -36., -32., -30., -28., -26.,    \
                -24., -22., -20., -18., -16., -14., -12., \
                -10.,  -9.,  -8.,  -7.,  -6.,  -5.,  -4. ]
elif (SimInstn == 2):    # (2), 60 comp_t_units
    δB_AXIS = [  -3.,  -2.,  -1.,   1. ]
elif (SimInstn == 3):    # (3), 50 comp_t_units
   δB_AXIS = [   2.,   3.  ]
elif (SimInstn == 4):    # (4), 65 comp_t_units
   δB_AXIS = [   4.,   5.,   6.,   7.,   8. ]   
elif (SimInstn == 5):    # (5), 55 comp_t_units
   δB_AXIS = [   9.,   10.,  12.,  14.,  16.,  18.,  20.,  22.,  24. ]
elif (SimInstn == 6):    # (6), 56 comp_t_units
   δB_AXIS = [  26.,  28.,  30.,  32.,  34.,  36.,  38.,  \
                40.,  45.,  50.,  55.,  60.,  65.,  70.,  75.,  80.,  85.,  90.,  95.,  100.  ]  

#-------#
if (SimInstn == 11):    
    δB_AXIS = [ -1. ]
elif (SimInstn == 12):    
    δB_AXIS = [ 1. ]
elif (SimInstn == 13):    
    δB_AXIS = [ 2. ]
elif (SimInstn == 14):    
    δB_AXIS = [ 3. ]
    
#-------#
N_ptsδB = len(δB_AXIS)   
#-------# Arrays to save info about collisions           
#array Ncoll_avg_e[N_ptsδB]; array Ncoll_avg_e2[N_ptsδB]  # number of collisions
Ncoll_avg = [0.0] * N_ptsδB
FLAG_Gmdt = [0] * N_ptsδB
Gammadt_max = [0.0] * N_ptsδB
Gammadt_avg = [0.0] * N_ptsδB
#--------#
#------------------------------------#

#--------#
#----------------------#
#------------------------------------#
#--------------------------------------------------#  [[[BEGIN Detuning cycle]]]
for jB in range(N_ptsδB):

    #--------# Optimize dt_step  # NOT ADAPTED YET
    dt_step  = DTSTEP[IsCloseToRes_dt(δB_AXIS[jB])]
    N_tsteps = int(t_final/dt_step + 1.e-5) + 1  # including first step @ t=0
    # Andi: we save each Nstep (t_05 seconds) the coords to file.
    #       this way we do not need to memorize all steps
    Nstep = int(round(t_05 / dt_step))
    print('\n%s\nNLi = %i, B = %.3f mG, time steps = %i, dt = %.3f us ...\n' % ('using vectors' if use_vectors else '', N_atoms, δB_AXIS[jB], N_tsteps, dt_step*1e6))

    if False:
        #-------# EnergyCons & coords (now GLOBAL) 
        with open(EnergyCons_det(SimSet,jB), 'w') as energycons:
            print("Det  (mG) \t AtomIndx \t Etot_in (J) \t Etot_fin (J) \t N_coll", file=energycons)
        
    #with open(Coords05_det(SimSet,jB), 'w') as coords05:
    with open(coords_file, 'w') as coords05:
        print(coords_hdr, file=coords05)
    
    #with open(MeanFreePath_det(SimSet,jB), 'w') as mfp:
    with open(mfp_file, 'w') as mfp:
        print(mfp_hdr, file=mfp)
    
    #--------# Generate initial distributions (re-do for every detuning)   --> can generate all at once?
    
    if use_vectors:
        two_pi = 2*np.pi
        
        rng = np.random.default_rng(seed=seed)
        rnd_gauss = rng.normal(loc=0, scale=1.0, size=N_atoms*6) # Gauss with mu=0, sigma=1
        X_in        = rnd_gauss[0:N_atoms*6:6]*SX0_Li    # center = 0
        Y_in        = rnd_gauss[1:N_atoms*6:6]*SY0_Li  
        Z_in        = rnd_gauss[2:N_atoms*6:6]*SZ0_Li
        Vx_in       = rnd_gauss[3:N_atoms*6:6]*SV0_Li 
        Vy_in       = rnd_gauss[4:N_atoms*6:6]*SV0z_Li
        Vz_in       = rnd_gauss[5:N_atoms*6:6]*SV0z_Li
        Ncoll_atom  = np.zeros((N_atoms,), dtype=np.uint64)
        Etot_in     = np.zeros((N_atoms,), dtype=np.float64) 
        Etot_fin    = np.zeros((N_atoms,), dtype=np.float64)
        AnalAx      = np.zeros((N_atoms,), dtype=np.float64)    
        AnalAy      = np.zeros((N_atoms,), dtype=np.float64)  
        AnalAz      = np.zeros((N_atoms,), dtype=np.float64)
        Analφx      = np.zeros((N_atoms,), dtype=np.float64)     
        Analφy      = np.zeros((N_atoms,), dtype=np.float64) 
        Analφz      = np.zeros((N_atoms,), dtype=np.float64)
        
        if seed is not None:
            # for easier comparison save transposed files as in old code
            # data_mfp is -1 for first entry (Δt_free) where no collision happened
            data_coords = np.empty((N_05+1, 9, N_atoms), dtype=np.float64)
            data_mfp    = np.ones((N_tsteps, 5, N_atoms), dtype=np.float64)*-1
    else:
        X_in  = [None] * N_atoms;  Y_in = [None] * N_atoms;  Z_in = [None] * N_atoms; 
        Vx_in = [None] * N_atoms; Vy_in = [None] * N_atoms; Vz_in = [None] * N_atoms; 
        
        if seed is None:
            for ja in range(N_atoms):
                X_in[ja]  = random.gauss(0.0, SX0_Li)    # center = 0
                Y_in[ja]  = random.gauss(0.0, SY0_Li)  
                Z_in[ja]  = random.gauss(0.0, SZ0_Li)
                Vx_in[ja] = random.gauss(0.0, SV0_Li) 
                Vy_in[ja] = random.gauss(0.0, SV0z_Li)
                Vz_in[ja] = random.gauss(0.0, SV0z_Li)
        else:
            # use same random number generator and precalculate all numbers for entire run
            # this is needed since in old code outer loop is over atoms
            # while in new code outer loop is over time
            # the exact order of calls and number and type of random numbers is important here!
            # order of index in lists is [jt][ja]
            rng = np.random.default_rng(seed=seed)
            rnd_gauss = rng.normal(loc=0, scale=1.0, size=N_atoms*6) # Gauss with mu=0, sigma=1
            for ja in range(N_atoms):
                X_in[ja]  = rnd_gauss[0+ja*6]*SX0_Li    # center = 0
                Y_in[ja]  = rnd_gauss[1+ja*6]*SY0_Li  
                Z_in[ja]  = rnd_gauss[2+ja*6]*SZ0_Li
                Vx_in[ja] = rnd_gauss[3+ja*6]*SV0_Li 
                Vy_in[ja] = rnd_gauss[4+ja*6]*SV0z_Li
                Vz_in[ja] = rnd_gauss[5+ja*6]*SV0z_Li

            if (MeanFieldDima == 1):
                Cr_samples_mf = np.empty((N_tsteps, N_atoms*3), dtype=np.float64)
            if (modecoll == 'Cr_RealT'):
                Cr_samples_real = np.empty((N_tsteps, N_atoms*3), dtype=np.float64)
            randcoll_all = np.empty((N_tsteps, N_atoms), dtype=np.float64)
            # note: the two following random numbers would need to be calculated only for colliding atoms
            #       but since we do not know this before the simulation we have to calculate them for all atoms
            #       and then use in both simulations with use_vectors = True/False.
            randcosTheta_all = np.empty((N_tsteps, N_atoms), dtype=np.float64)
            randPhi_all      = np.empty((N_tsteps, N_atoms), dtype=np.float64)
            # note: jt=0 is left uninitialized but this way indexing is more intuitive
            for jt in range(1, N_tsteps):
                if (MeanFieldDima == 1):
                    #----# Generate gaussrand Cr velocity  (independent from collision one)      
                    Cr_samples_mf[jt] = rng.normal(loc=0, scale=1.0, size=N_atoms*3) # Gauss with mu=0, sigma=1         

                if (modecoll == 'Cr_RealT'):
                    #----# Generate gaussrand Cr velocity  
                    Cr_samples_real[jt] = rng.normal(loc=0, scale=1.0, size=N_atoms*3) # Gauss with mu=0, sigma=1         

                randcoll_all[jt] = rng.uniform(low=0.0, high=1.0, size=N_atoms) # uniform distribution

                randcosTheta_all[jt] = rng.uniform(-1.,1.,size=N_atoms)     # = z projection, flat
                randPhi_all[jt]      = rng.uniform(0., 2*π, size=N_atoms)   # This one instead is fine, indeed is actually good.    

        #--------# Declare some arrays (re-initialized at every detuning)
        Ncoll_atom = [0] * N_atoms
        Etot_in    = [None] * N_atoms; Etot_fin = [None] * N_atoms
        AnalAx = [None] * N_atoms;     AnalAy = [None] * N_atoms;  AnalAz = [None] * N_atoms   # consider not having them as arrays/lists
        Analφx = [None] * N_atoms;     Analφy = [None] * N_atoms;  Analφz = [None] * N_atoms   # but as simple variables

    #--------# Optimize dt_step  # NOT ADAPTED YET
    #dt_step  = DTSTEP[IsCloseToRes_dt(δB_AXIS[jB])]
    #N_tsteps = int(t_final/dt_step + 1.e-5) + 1  # including first step @ t=0
    #--------# 
    #--------#
    #--------# NB: NEW POSITION OPEN MFP   (!!!)
    #with open(MeanFreePath_det(SimSet,jB), 'a') as mfp:
    with open(mfp_file, 'a') as mfp:
        #with open(Coords05_det(SimSet,jB), 'a') as coords05:
        with open(coords_file, 'a') as coords05:
            #--------------------------------#  (((BEGIN:   Atoms/Histories cycle)))

            if use_vectors:
                #jt_prevcoll = 0     # python arrays start from 0   
                tx = ωxLi*X_in
                ty = ωyLi*Y_in
                tz = ωzLi*Z_in
                Etot_in = 0.5*m_Li6*(Vx_in*Vx_in + Vy_in*Vy_in + Vz_in*Vz_in)  +  \
                          0.5*m_Li6*(tx*tx + ty*ty + tz*tz)

                #-----# Declare & initialize trajectory arrays 
                # Andi: we just need one vector to propagate the state of the atoms each step.
                #       Xt_prevcoll is the x-coordinate of the last collisions used for Δx_free
                #       no other coordinates are needed.
                # note: by assigning Xt_ = X_in we do not copy but assign 'pointer' Xt_ 
                #       to the memory location of X_in. i.e. when we change Xt_ later,
                #       this will change also X_in!
                #       if this is not what you want, copy() explicitly as done with Xt_prevcoll.
                jt_prevcoll = np.zeros((N_atoms,), dtype=np.uint64)
                Xt_prevcoll = X_in.copy()
                RealΔx_free = np.zeros((N_atoms,), dtype=np.float64)
                Real_λ_free = np.zeros((N_atoms,), dtype=np.float64)
                Xt_         = X_in
                Yt_         = Y_in
                Zt_         = Z_in
                Vxt_        = Vx_in
                Vyt_        = Vy_in
                Vzt_        = Vz_in
                #-----# Initialize A and φ arrays for analmode (t=0)
                Analφx = np.arctan2(ωxLi*X_in, Vx_in)%two_pi      # atan2 is def btw -π and +π
                Analφy = np.arctan2(ωyLi*Y_in, Vy_in)%two_pi      # trick: keep atan2 here, and mod2pi only after collisions
                Analφz = np.arctan2(ωzLi*Z_in, Vz_in)%two_pi      #        then you get a timescale for dephasing
                AnalAx = Vx_in/(ωxLi*np.cos(Analφx))  
                AnalAy = Vy_in/(ωyLi*np.cos(Analφy))  
                AnalAz = Vz_in/(ωzLi*np.cos(Analφz))  

                # initial conditions
                timejt = 0
                #-----# Print (Coords05)  --> NB:  Xt_,... arrays are OVERWRITTEN!  but opening can be moved
                #jmult05_ = jCorrespTo_time(t05_from_j05(j05))
                if seed is None:
                    if save_text:
                        np.savetxt(coords05, np.transpose([
                            [δB_AXIS[jB]]*N_atoms, np.arange(N_atoms)+1, [timejt*1e3]*N_atoms,
                            1e6*Xt_, 1e6*Yt_, 1e6*Zt_, 1e3*Vxt_, 1e3*Vyt_, 1e3*Vzt_]), 
                            fmt=coords_fmt, delimiter=sep)
                    else:
                        for ja in range(N_atoms):
                            print(coords_fmt_all % (δB_AXIS[jB], ja+1, timejt*1e3,
                                1e6*Xt_[ja], 1e6*Yt_[ja], 1e6*Zt_[ja], 1e3*Vxt_[ja], 1e3*Vyt_[ja], 1e3*Vzt_[ja]
                                ), file=coords05)
                else:
                    data_coords[0] = [[δB_AXIS[jB]]*N_atoms, np.arange(N_atoms)+1, [timejt*1e3]*N_atoms,
                            1e6*Xt_, 1e6*Yt_, 1e6*Zt_, 1e3*Vxt_, 1e3*Vyt_, 1e3*Vzt_]

                #-----------------------# [[[ BEGIN Trajectory cycle]]]  (dt_steps)
                for jt in range(1, N_tsteps): 

                    # Andi: some notes on optimization which matter when doing many loops!
                    # - avoid function calls when not needed
                    # - generation of N_atoms random numbers at once is much more efficient that calling 
                    #   N_atoms times random.normal or random.uniform.
                    # - avoid sqrt and power when not absolutely needed. these are costly functions!
                    #   - use x*x for powers of 2 and x*x*x for powers of 3.
                    #   - do not use sumq3()**2
                    # - precalculate all constants like hbar/m_red
                    #   I usually try also to get rid of large powers by multiplying both sides 
                    #   of an equation with a constant, which then can be absorbed into the other constants.
                    #   this avoids numerical problems when calculations involve many orders of magnitude.
                    #   however, with np.float64 (aka 'float' in python) this is not a too big problem in the present case.
                    # - copying is costly but often cannot be avoided, however sometimes it can be avoided:
                    #   b = [ai for ai in a[::3]] this copies all 3rd elements of a into b
                    #   b = a[::3] takes all 3rd elements of a but without copying.
                    # - some constructs are more efficient: 
                    #   a = a + b    copies the sum into a new memory which then gets assigned to a.
                    #   a += b       directly copies the sum into the old memory a. this is giving only a small gain.
                    #   a[coll] += b this might give a larger gain since has to select 'coll' only once
                    timejt = dt_step*jt
                    ωx_t = ωxLi*timejt
                    ωy_t = ωyLi*timejt
                    ωz_t = ωzLi*timejt
                    
                    # old position
                    tx = Xt_
                    ty = Yt_ 
                    tz = Zt_

                    #----------# Propagation for one dt_step  (analytic harmonic oscillator)
                    Xt_ = AnalAx * np.sin(ωx_t + Analφx)
                    Yt_ = AnalAy * np.sin(ωy_t + Analφy)
                    Zt_ = AnalAz * np.sin(ωz_t + Analφz)

                    # position change
                    tx -= Xt_
                    ty -= Yt_
                    tz -= Zt_

                    Vxt_ = (ωxLi*AnalAx) * np.cos(ωx_t + Analφx)
                    Vyt_ = (ωyLi*AnalAy) * np.cos(ωy_t + Analφy)
                    Vzt_ = (ωzLi*AnalAz) * np.cos(ωz_t + Analφz)

                    if test_jt == jt:
                        ja = test_ja
                        print('Li vec(0)', jt, ja, Xt_[ja], Yt_[ja], Zt_[ja], Vxt_[ja], Vyt_[ja], Vzt_[ja])

                    # Andi: integrate MFP for all atoms in each step. 
                    #       this way we do not need to save coordinates of each step.
                    #Δt_free        = (1.e3) * (jt - jt_prevcoll) * dt_step     # [ms]
                    #Δx_free        = (1.e6) * m.fabs(Xt_[jt] - Xt_[jt_prevcoll])  # [µm]
                    #RealΔx_free    = (1.e6) * sum([(m.fabs(Xt_[qx+1] - Xt_[qx])) for qx in range(jt_prevcoll, jt)])   # [µm]   
                    #Real_λ_free    = (1.e6) * sum([(sumq3((Xt_[qx+1]-Xt_[qx]), (Yt_[qx+1]-Yt_[qx]), (Zt_[qx+1]-Zt_[qx]))) for qx in range(jt_prevcoll, jt)])  # [µm]     
                    RealΔx_free += (1.e6) * np.abs(tx)   # [µm]   
                    Real_λ_free += (1.e6) * np.sqrt(tx*tx + ty*ty + tz*tz)  # [µm]     

                    #----------# Mean field à la Dima:
                    if (MeanFieldDima == 1):
                        #----# Generate gaussrand Cr velocity  (independent from collision one)      
                        samples = rng.normal(loc=0, scale=1.0, size=N_atoms*3) # Gauss with mu=0, sigma=1         
                        VxCr   = samples[0:N_atoms*3:3]*SV0x_Cr
                        VyCr   = samples[1:N_atoms*3:3]*SV0_Cr
                        VzCr   = samples[2:N_atoms*3:3]*SV0_Cr
                        #----# Relative velocity
                        Vrelx  = Vxt_ - VxCr  
                        Vrely  = Vyt_ - VyCr  
                        Vrelz  = Vzt_ - VzCr   
                        Vrelt  = np.sqrt(Vrelx*Vrelx + Vrely*Vrely + Vrelz*Vrelz)
                        #----# Velocity change 
                        Vxt_ -= dEMF_dx(Xt_,Yt_,Zt_, (m_red*Vrelt/h_bar), δB_AXIS[jB])/m_Li6 * dt_step
                        Vyt_ -= dEMF_dy(Xt_,Yt_,Zt_, (m_red*Vrelt/h_bar), δB_AXIS[jB])/m_Li6 * dt_step
                        Vzt_ -= dEMF_dz(Xt_,Yt_,Zt_, (m_red*Vrelt/h_bar), δB_AXIS[jB])/m_Li6 * dt_step
                        #----# Re-compute motion parameters 
                        Analφx = ( np.atan2((ωxLi*Xt_), Vxt_) - ωx_t )%two_pi
                        Analφy = ( np.atan2((ωyLi*Yt_), Vyt_) - ωy_t )%two_pi
                        Analφz = ( np.atan2((ωzLi*Zt_), Vzt_) - ωz_t )%two_pi
                        AnalAx = Vxt_/(ωxLi*np.cos(ωx_t + Analφx))
                        AnalAy = Vyt_/(ωyLi*np.cos(ωy_t + Analφy))
                        AnalAz = Vzt_/(ωzLi*np.cos(ωz_t + Analφz))  
            
                    #----------# {{{ Collisions }}}                    
                    if (modecoll == 'Cr_RealT'):
                        #----# Generate gaussrand Cr velocity  
                        samples = rng.normal(loc=0, scale=1.0, size=N_atoms*3) # Gauss with mu=0, sigma=1         
                        VxCr   = samples[0:N_atoms*3:3]*SV0x_Cr
                        VyCr   = samples[1:N_atoms*3:3]*SV0_Cr
                        VzCr   = samples[2:N_atoms*3:3]*SV0_Cr
                        #----# Relative velocity  (before collision)
                        Vrelx  = Vxt_ - VxCr  
                        Vrely  = Vyt_ - VyCr  
                        Vrelz  = Vzt_ - VzCr   
                        Vrelt  = np.sqrt(Vrelx*Vrelx + Vrely*Vrely + Vrelz*Vrelz)
                        #----#
                        Gamma_dt = Γel_vrel(δB_AXIS[jB], Vrelt, n_Gauss(crbath*Xt_,crbath*Yt_,crbath*Zt_, Natom_Cr,sx_Cr,sy_Cr,sz_Cr)) * dt_step
                    else: #if (modecoll == 'Cr_M∞'):
                        tx = Vxt_
                        ty = Vyt_
                        tz = Vzt_
                        VLitot_  = np.sqrt(tx*tx + ty*ty + tz*tz)  
                        Gamma_dt = Γel_vLi(δB_AXIS[jB], VLitot_, n_Gauss(crbath*Xt_,crbath*Yt_,crbath*Zt_, Natom_Cr,sx_Cr,sy_Cr,sz_Cr)) * dt_step  
                    
                    #-----#
                    Gammadt_avg[jB] += np.sum(Gamma_dt)
                    #if (Gamma_dt > 1.0):
                    #    FLAG_Gmdt[jB] = FLAG_Gmdt[jB] + 1     # check that Gamma*dt is less than 1
                    FLAG_Gmdt[jB] += np.count_nonzero(Gamma_dt > 1.0)     # check that Gamma*dt is less than 1
                    
                    #if (Gamma_dt > Gammadt_max[jB]):
                    #    Gammadt_max[jB] = Gamma_dt 
                    tx = np.max(Gamma_dt)
                    if tx > Gammadt_max[jB]:
                        Gammadt_max[jB] = tx
                    
                    if test_jt == jt:
                        ja = test_ja
                        print('Cr vec(0)', jt, ja, VxCr[ja], VyCr[ja], VzCr[ja]) #, Gammadt_avg[jB], FLAG_Gmdt[jB], Gammadt_max[jB])

                    #-----# COLLISION STUFF

                    # Andi: instead of checking if (randcoll < Gamma_dt) for each atom,
                    #       we define array of True/False values 'coll' (often labelled 'mask'). 
                    #       this is True for colliding atoms and False for non-colliding atoms.
                    #       indexing with 'coll' allows to work only on the sub-set of colliding atoms.
                    #       this has some overhead of selecting these atoms and writing back.
                    #       so ideally we should select these atoms, work on them, 
                    #       and write back only once the end result into the array of all atoms.
                    randcoll = rng.uniform(low=0.0, high=1.0, size=N_atoms) # uniform distribution
                    coll = (randcoll < Gamma_dt)   #  --> ((Collision happening!))   
                    #if (randcoll < Gamma_dt):   #  --> ((Collision happening!))
                    N_coll = np.count_nonzero(coll) # number of colliding atoms 
                    
                    if test_jt == jt:
                        ja = test_ja
                        print(Ncoll_atom[ja], randcoll[ja], '<', Gamma_dt[ja], coll[ja])
                    
                    #---# Update counters, calculate MFP stuff
                    Ncoll_atom[coll] += 1
                    #Δt_free        = (1.e3) * (jt - jt_prevcoll) * dt_step     # [ms]
                    #Δx_free        = (1.e6) * m.fabs(Xt_[jt] - Xt_[jt_prevcoll])  # [µm]
                    Δt_free        = (1.e3) * (jt - jt_prevcoll[coll])* dt_step     # [ms]
                    Δx_free        = (1.e6) * np.abs(Xt_[coll] - Xt_prevcoll[coll])  # [µm]
                    #RealΔx_free    = (1.e6) * sum([(m.fabs(Xt_[qx+1] - Xt_[qx])) for qx in range(jt_prevcoll, jt)])   # [µm]   
                    #Real_λ_free    = (1.e6) * sum([(sumq3((Xt_[qx+1]-Xt_[qx]), (Yt_[qx+1]-Yt_[qx]), (Zt_[qx+1]-Zt_[qx]))) for qx in range(jt_prevcoll, jt)])  # [µm]     
                    krel_RetTime   = (m_red*Vrelt[coll]/h_bar) if (modecoll == 'Cr_RealT') else (m_Li6*VLitot_[coll]/h_bar)
                    Retard_Time    = RetardTime(δB_AXIS[jB], krel_RetTime)     # [s], but then saved in [µs]
                    ωx_Rt = ωxLi*Retard_Time
                    ωy_Rt = ωyLi*Retard_Time
                    ωz_Rt = ωzLi*Retard_Time
                    #---# OPENING MOVED OUTSIDE ATOM CYCLE (!!!)
                    #with open(MeanFreePath_det(SimSet,jB), 'a') as mfp:
                    #    print(Δt_free, Δx_free, RealΔx_free, Real_λ_free, (1.e6)*Retard_Time, file=mfp)
                    if save_mfp:
                        ax = RealΔx_free[coll]
                        ay = Real_λ_free[coll]
                        if seed is None:
                            if save_text:
                                np.savetxt(mfp, np.transpose([Δt_free, Δx_free, ax, ay, (1.e6)*Retard_Time]), fmt=mfp_fmt, delimiter=sep)
                            else:
                                for ja in range(N_coll):
                                    print(mfp_fmt_all % (Δt_free[ja], Δx_free[ja], ax[ja], ay[ja], (1.e6)*Retard_Time[ja]), file=mfp)
                        else:
                            # note: somehow have to transpose here which is unexpected?
                            data_mfp[jt,:,coll] = np.transpose([Δt_free, Δx_free, ax, ay, 1e6*Retard_Time])
                        
                    # reset MFP for colliding atoms only
                    jt_prevcoll[coll] = jt
                    Xt_prevcoll[coll] = Xt_[coll]
                    RealΔx_free[coll] = 0
                    Real_λ_free[coll] = 0

                    #---# s-wave collision (now good!)
                    #randTheta     = RandNum(0., π)     # ((NB)):  WRONG!!
                    if seed is None:
                        randcosTheta   = rng.uniform(-1.,1.,size=N_coll)    # = z projection, flat
                        randTheta      = np.arccos(randcosTheta)
                        randPhi        = rng.uniform(0., 2*π, size=N_coll)   # This one instead is fine, indeed is actually good.    
                    else:
                        randcosTheta   = rng.uniform(-1.,1.,size=N_atoms)    # = z projection, flat
                        randTheta      = np.arccos(randcosTheta)
                        randPhi        = rng.uniform(0., 2*π, size=N_atoms)   # This one instead is fine, indeed is actually good.    
                        if test_jt == jt and coll[test_ja]:
                            print('theta', randcosTheta[test_ja], 'fi', randPhi[test_ja]) 
                        randTheta = randTheta[coll]
                        randPhi   = randPhi  [coll]

                    if (modecoll == 'Cr_RealT'):    # Cr has FINITE mass and a Temperature, it DOES recoil and CAN transfer/absorb energy from Li 
                        #----# After collision
                        tmp = Vrelt[coll]
                        Vrelx_ = tmp * np.sin(randTheta)*np.cos(randPhi)
                        Vrely_ = tmp * np.sin(randTheta)*np.sin(randPhi)
                        Vrelz_ = tmp * np.cos(randTheta)
                        #----# Li exit velocity  (OVERWRITE  Vxt_[jt])
                        Vxt_[coll] = vx = (Vxt_[coll] + MassRat*(VxCr[coll]+Vrelx_)) / (1.+MassRat)    # checked, requiring momentum conservation
                        Vyt_[coll] = vy = (Vyt_[coll] + MassRat*(VyCr[coll]+Vrely_)) / (1.+MassRat)
                        Vzt_[coll] = vz = (Vzt_[coll] + MassRat*(VzCr[coll]+Vrelz_)) / (1.+MassRat)
                        #----#
                    else: #if (modecoll == 'Cr_M∞'):       # Cr has infinite mass, it does not recoil nor transfer/absorb energy from Li, AND IT'S NOT MOVING AT ALL
                        tmp = VLitot_[coll]
                        Vxt_[coll]  = vx = tmp * np.sin(randTheta)*np.cos(randPhi)
                        Vyt_[coll]  = vy = tmp * np.sin(randTheta)*np.sin(randPhi)
                        Vzt_[coll]  = vz = tmp * np.cos(randTheta)             
                    
                    if test_jt == jt:
                        ja = test_ja
                        print('Li vec(1)', jt, ja, Vxt_[ja], Vyt_[ja], Vzt_[ja]) #, Gammadt_avg[jB], FLAG_Gmdt[jB], Gammadt_max[jB])

                    #-------# Re-compute motion parameters (if no RetTime was there)  (this is needed anyway)
                    Analφx[coll] = ax = ( np.arctan2((ωxLi*Xt_[coll]), vx) - ωx_t )%two_pi
                    Analφy[coll] = ay = ( np.arctan2((ωyLi*Yt_[coll]), vy) - ωy_t )%two_pi
                    Analφz[coll] = az = ( np.arctan2((ωzLi*Zt_[coll]), vz) - ωz_t )%two_pi
                    AnalAx[coll] = vx/(ωxLi*np.cos(ωx_t + ax))
                    AnalAy[coll] = vy/(ωyLi*np.cos(ωy_t + ay))
                    AnalAz[coll] = vz/(ωzLi*np.cos(ωz_t + az))

                    #-------# Retard time correction on phase (after having computed amplitude)  --> SHIFT ON LI
                    if (RetTimePhase == 1):
                        Analφx[coll] = ax = ( ax - ωx_Rt )%two_pi
                        Analφy[coll] = ay = ( ay - ωx_Rt )%two_pi
                        Analφz[coll] = az = ( az - ωx_Rt )%two_pi
                    
                    #-------# Retard time correction on RELATIVE PARTICLE + COM
                    if (RetTimePhase == 2):
                        #----# Cr output velocity (if no ret.time).  NB:  Vxt_[jt] at this stage is ALREADY the Li output velocity
                        
                        # Andi note: Vrelx/y/z_ are defined only if modecoll == 'Cr_RealT'
                        
                        VxCr_ = vx - Vrelx_
                        VyCr_ = vy - Vrely_
                        VzCr_ = vz - Vrelz_
                        #----# C.o.M. velocity  (before AND after collision)  [calculated with params AFTER collision IF NO ret time]
                        # Not needed in the end
                        #Vcomx  = (m_Li6*Vxt_[jt] + m_Cr53*VxCr_)/m_LiCr
                        #Vcomy  = (m_Li6*Vyt_[jt] + m_Cr53*VyCr_)/m_LiCr
                        #Vcomz  = (m_Li6*Vzt_[jt] + m_Cr53*VzCr_)/m_LiCr
                        #Vcomt  = sumq3(Vcomx, Vcomy, Vcomz)
                        #----# Li "unperturbed" position/velocity at time (t* + Δt_ret)  [for c.o.m.]
                        tx = ωx_t + ωx_Rt + ax
                        ty = ωy_t + ωy_Rt + ay
                        tz = ωz_t + ωz_Rt + az
                        ax = AnalAx[coll]
                        ay = AnalAy[coll]
                        az = AnalAz[coll]
                        xLi__  = ax * np.sin(tx)
                        yLi__  = ay * np.sin(ty)
                        zLi__  = az * np.sin(tz)
                        vxLi__ = (ωxLi*ax) * np.cos(tx)
                        vyLi__ = (ωyLi*ay) * np.cos(ty)
                        vzLi__ = (ωzLi*az) * np.cos(tz)
                        #----# Cr motion params (after collision, if no retard time)     [for c.o.m.]
                        tx = ωxCr*timejt
                        ty = ωyCr*timejt
                        tz = ωzCr*timejt
                        Crφx_  = ( np.atan2((ωxCr*Xt_[coll]), VxCr_) - tx )%two_pi
                        Crφy_  = ( np.atan2((ωyCr*Yt_[coll]), VyCr_) - ty )%two_pi
                        Crφz_  = ( np.atan2((ωzCr*Zt_[coll]), VzCr_) - tz )%two_pi
                        CrAx_  = VxCr_/(ωxCr*np.cos(tx + Crφx_))
                        CrAy_  = VyCr_/(ωyCr*np.cos(ty + Crφy_))
                        CrAz_  = VzCr_/(ωzCr*np.cos(tz + Crφz_))
                        #----# Cr "unperturbed" position/velocity at time (t* + Δt_ret)  [for c.o.m.]
                        tx += ωxCr*Retard_Time + Crφx_
                        ty += ωyCr*Retard_Time + Crφy_
                        tz += ωzCr*Retard_Time + Crφz_
                        xCr__  = CrAx_ * np.sin(tx)
                        yCr__  = CrAy_ * np.sin(ty)
                        zCr__  = CrAz_ * np.sin(tz)
                        vxCr__ = (ωxCr*CrAx_) * np.cos(tx)
                        vyCr__ = (ωyCr*CrAy_) * np.cos(ty)
                        vzCr__ = (ωzCr*CrAz_) * np.cos(tz)
                        #----# C.o.M. "unperturbed" position/velocity at time (t* + Δt_ret)
                        xcom__ = WeightAvg2(xLi__,m_Li6, xCr__,m_Cr53)
                        ycom__ = WeightAvg2(yLi__,m_Li6, yCr__,m_Cr53)
                        zcom__ = WeightAvg2(zLi__,m_Li6, zCr__,m_Cr53)
                        vxcom__= WeightAvg2(vxLi__,m_Li6, vxCr__,m_Cr53)
                        vycom__= WeightAvg2(vyLi__,m_Li6, vyCr__,m_Cr53)
                        vzcom__= WeightAvg2(vzLi__,m_Li6, vzCr__,m_Cr53)
                        #----# Fix passage at time (t* + Δt_ret) in new Li trajectory
                        xListar  = xcom__  # relative distance = 0
                        yListar  = ycom__
                        zListar  = zcom__
                        VxListar = vxcom__ + MassRat/(1.+MassRat) * Vrelx_
                        VyListar = vycom__ + MassRat/(1.+MassRat) * Vrely_
                        VzListar = vzcom__ + MassRat/(1.+MassRat) * Vrelz_
                        #----# New Motion params (final ones)
                        tx = ωx_t + ωx_Rt
                        ty = ωy_t + ωy_Rt
                        tz = ωz_t + ωz_Rt
                        Analφx[coll] = ax = ( np.arctan2((ωxLi*xListar), VxListar) - tx )%two_pi
                        Analφy[coll] = ay = ( np.arctan2((ωyLi*yListar), VyListar) - ty )%two_pi
                        Analφz[coll] = az = ( np.arctan2((ωzLi*zListar), VzListar) - tz )%two_pi
                        AnalAx[coll] = VxListar/(ωxLi*np.cos(tx + ax))
                        AnalAy[coll] = VyListar/(ωyLi*np.cos(ty + ay))
                        AnalAz[coll] = VzListar/(ωzLi*np.cos(tz + az))
                        #----#
                    
                    if (test_jt == jt) and coll[test_ja]:
                        ja = test_ja
                        print('Li vec(2)', jt, ja, Analφx[ja], Analφy[ja], Analφz[ja], AnalAx[ja], AnalAy[ja], AnalAz[ja])

                    if (jt % Nstep) == 0:
                        # save coordinates every t_05 seconds:
                        # this is now transposed, i.e. fast changing axis is atoms, before was time.
                        #-----# Print (Coords05)  --> NB:  Xt_,... arrays are OVERWRITTEN!  but opening can be moved
                        #jmult05_ = jCorrespTo_time(t05_from_j05(j05))
                        if seed is None:
                            if save_text:
                                np.savetxt(coords05, np.transpose([
                                    [δB_AXIS[jB]]*N_atoms, np.arange(N_atoms)+1, [timejt*1e3]*N_atoms,
                                        1e6*Xt_, 1e6*Yt_, 1e6*Zt_, 1e3*Vxt_, 1e3*Vyt_, 1e3*Vzt_]), 
                                    fmt=coords_fmt, delimiter=sep)
                            else:
                                for ja in range(N_atoms):
                                    print(coords_fmt_all % (δB_AXIS[jB], ja+1, timejt*1e3,
                                        1e6*Xt_[ja], 1e6*Yt_[ja], 1e6*Zt_[ja], 1e3*Vxt_[ja], 1e3*Vyt_[ja], 1e3*Vzt_[ja]
                                        ), file=coords05)
                        else:
                            data_coords[int(jt//Nstep)] = [[δB_AXIS[jB]]*N_atoms, np.arange(N_atoms)+1, [timejt*1e3]*N_atoms,
                                1e6*Xt_, 1e6*Yt_, 1e6*Zt_, 1e3*Vxt_, 1e3*Vyt_, 1e3*Vzt_]
 

                #-----------------------# [[[ END Trajectory cycle]]]
                #-----------------------#
                #-----------------------#
                #-----# Calculate (EnergyCons)   ---> print moved outside ATOM cycle, before END DET cycle. E_tot arrays are NOT overwritten
                tx = ωxLi*Xt_
                ty = ωyLi*Yt_
                tz = ωzLi*Zt_
                Etot_fin = 0.5*m_Li6*( (Vxt_*Vxt_ + Vyt_*Vyt_ + Vzt_*Vzt_)  +  \
                                       (tx*tx + ty*ty + tz*tz) )
                #with open(EnergyCons_det(SimSet,jB), 'a') as energycons:
                #    print(δB_AXIS[jB], ja+1, Etot_in[ja], Etot_fin[ja], Ncoll_atom[ja], file=energycons)
                                
                if seed is not None:
                    # for easier comparison save transposed files as in old code
                    # data_mfp is -1 for first entry (Δt_free) where no collision happened
                    # note: np.savetxt should work without loops but needs a bit of testing 
                    #       since have to transpose right axes and select colliding atoms.
                    for ja in range(N_atoms):
                        for jt in range(0,N_tsteps):
                            if (jt % Nstep) == 0:
                                print(coords_fmt_all % tuple(data_coords[int(jt//Nstep),m,ja] for m in range(9)), file=coords05)
                            if data_mfp[jt,0,ja] != -1:
                                # save mfp of colliding atom
                                print(mfp_fmt_all % tuple(data_mfp[jt,m,ja] for m in range(5)), file=mfp)

                #for ja in range(N_atoms):
                #    #-----# Print (Coords05)  --> NB:  Xt_,... arrays are OVERWRITTEN!  but opening can be moved
                #    #with open(Coords05_det(SimSet,jB), 'a') as coords05:
                #    for j05 in range(N_05+1):
                #        jmult05_ = jCorrespTo_time(t05_from_j05(j05))
                #        print( δB_AXIS[jB], ja+1, t05_from_j05(j05), \
                #            (1.e6*Xt_[jmult05_,ja]), (1.e6*Yt_[jmult05_,ja]), (1.e6*Zt_[jmult05_,ja]), \
                #            (1.e3*Vxt_[jmult05_,ja]), (1.e3*Vyt_[jmult05_,ja]), (1.e3*Vzt_[jmult05_,ja]), file=coords05)
                            
                #----# add spaces if un-comment with open
                #-----# λ_free for zero collisions (& last point anyway)
                # if (jt_prevcoll != (N_tsteps-1) and SaveLastPoint):
                if SaveLastPoint:
                    coll = (jt_prevcoll != (N_tsteps-1))
                    #Δt_free        = (1.e3) * ((N_tsteps-1) - jt_prevcoll) * dt_step
                    #Δx_free        = (1.e6) * m.fabs(Xt_[N_tsteps-1,ja] - Xt_[jt_prevcoll,ja])
                    Δt_free        = (1.e3) * ((N_tsteps-1) - jt_prevcoll[coll])* dt_step     # [ms]
                    Δx_free        = (1.e6) * np.abs(Xt_[coll] - Xt_prevcoll[coll])  # [µm]
                    #RealΔx_free    = (1.e6) * sum([(m.fabs(Xt_[qx+1,ja] - Xt_[qx,ja])) for qx in range(jt_prevcoll, N_tsteps-1)])
                    #Real_λ_free    = (1.e6) * sum([(sumq3((Xt_[qx+1,ja]-Xt_[qx,ja]), (Yt_[qx+1,ja]-Yt_[qx,ja]), (Zt_[qx+1,ja]-Zt_[qx,ja]))) for qx in range(jt_prevcoll, N_tsteps-1)])
                    Retard_Time    = 0.0
                    #----# Commented out for a while
                    #with open(MeanFreePath_det(SimSet,jB), 'a') as mfp:
                    ##    print(Δt_free, Δx_free, RealΔx_free, Real_λ_free, Retard_Time, file=mfp)  
                    #    ax = RealΔx_free[coll]
                    #    ay = Real_λ_free[coll]
                    #    for ja in range(N_coll):
                    #        print(Δt_free[ja], Δx_free[ja], ax[ja], ay[ja], (1.e6)*Retard_Time[ja], file=mfp)

            else:
            
                # 'classical' loop over atoms
                for ja in range(N_atoms):  
                    jt_prevcoll = 0     # python arrays start from 0   
                    Etot_in[ja] = 0.5*m_Li6*sumq3(Vx_in[ja],Vy_in[ja],Vz_in[ja])**2  +  \
                                  0.5*m_Li6*sumq3((ωxLi*X_in[ja]), (ωyLi*Y_in[ja]), (ωzLi*Z_in[ja]))**2
                    #-----# Declare & initialize trajectory arrays 
                    Xt_  = [None] * N_tsteps;  Xt_[0]  = X_in[ja]
                    Yt_  = [None] * N_tsteps;  Yt_[0]  = Y_in[ja]
                    Zt_  = [None] * N_tsteps;  Zt_[0]  = Z_in[ja]
                    Vxt_ = [None] * N_tsteps;  Vxt_[0] = Vx_in[ja]
                    Vyt_ = [None] * N_tsteps;  Vyt_[0] = Vy_in[ja]
                    Vzt_ = [None] * N_tsteps;  Vzt_[0] = Vz_in[ja]
                    #-----# Initialize A and φ arrays for analmode (t=0)
                    Analφx[ja] = mod2π( m.atan2((ωxLi*X_in[ja]), Vx_in[ja]) )      # atan2 is def btw -π and +π
                    Analφy[ja] = mod2π( m.atan2((ωyLi*Y_in[ja]), Vy_in[ja]) )      # trick: keep atan2 here, and mod2pi only after collisions
                    Analφz[ja] = mod2π( m.atan2((ωzLi*Z_in[ja]), Vz_in[ja]) )      #        then you get a timescale for dephasing
                    AnalAx[ja] = Vx_in[ja]/(ωxLi*m.cos(Analφx[ja]))  
                    AnalAy[ja] = Vy_in[ja]/(ωyLi*m.cos(Analφy[ja]))  
                    AnalAz[ja] = Vz_in[ja]/(ωzLi*m.cos(Analφz[ja]))  
                                           
                    #-----------------------# [[[ BEGIN Trajectory cycle]]]  (dt_steps)
                    for jt in range(1, N_tsteps): 
                                        
                        #----------# Propagation for one dt_step  (analytic harmonic oscillator)
                        Xt_[jt]  = AnalAx[ja] * m.sin(ωxLi*timejt(jt) + Analφx[ja])
                        Yt_[jt]  = AnalAy[ja] * m.sin(ωyLi*timejt(jt) + Analφy[ja])
                        Zt_[jt]  = AnalAz[ja] * m.sin(ωzLi*timejt(jt) + Analφz[ja])
                        Vxt_[jt] = (ωxLi*AnalAx[ja]) * m.cos(ωxLi*timejt(jt) + Analφx[ja])
                        Vyt_[jt] = (ωyLi*AnalAy[ja]) * m.cos(ωyLi*timejt(jt) + Analφy[ja])
                        Vzt_[jt] = (ωzLi*AnalAz[ja]) * m.cos(ωzLi*timejt(jt) + Analφz[ja])
                        
                        if (test_jt == jt) and (test_ja == ja):
                            print('Li cls(0)', jt, ja, Xt_[jt], Yt_[jt], Zt_[jt], Vxt_[jt], Vyt_[jt], Vzt_[jt])
                        
                        #----------# Mean field à la Dima:
                        if (MeanFieldDima == 1):
                            #----# Generate gaussrand Cr velocity  (independent from collision one)      
                            if seed is None:
                                VxCr   = random.gauss(0.0, SV0x_Cr)
                                VyCr   = random.gauss(0.0, SV0_Cr)
                                VzCr   = random.gauss(0.0, SV0_Cr)
                            else:
                                VxCr   = Cr_samples_mf[jt,0+ja*3]*SV0x_Cr
                                VyCr   = Cr_samples_mf[jt,1+ja*3]*SV0_Cr
                                VzCr   = Cr_samples_mf[jt,2+ja*3]*SV0_Cr
                            #----# Relative velocity
                            Vrelx  = Vxt_[jt] - VxCr  
                            Vrely  = Vyt_[jt] - VyCr  
                            Vrelz  = Vzt_[jt] - VzCr   
                            Vrelt  = sumq3(Vrelx, Vrely, Vrelz)
                            #----# Velocity change 
                            Vxt_[jt] = Vxt_[jt] - dEMF_dx(Xt_[jt],Yt_[jt],Zt_[jt], (m_red*Vrelt/h_bar), δB_AXIS[jB])/m_Li6 * dt_step
                            Vyt_[jt] = Vyt_[jt] - dEMF_dy(Xt_[jt],Yt_[jt],Zt_[jt], (m_red*Vrelt/h_bar), δB_AXIS[jB])/m_Li6 * dt_step
                            Vzt_[jt] = Vzt_[jt] - dEMF_dz(Xt_[jt],Yt_[jt],Zt_[jt], (m_red*Vrelt/h_bar), δB_AXIS[jB])/m_Li6 * dt_step
                            #----# Re-compute motion parameters 
                            Analφx[ja] = mod2π( m.atan2((ωxLi*Xt_[jt]), Vxt_[jt]) - ωxLi*timejt(jt) )
                            Analφy[ja] = mod2π( m.atan2((ωyLi*Yt_[jt]), Vyt_[jt]) - ωyLi*timejt(jt) )
                            Analφz[ja] = mod2π( m.atan2((ωzLi*Zt_[jt]), Vzt_[jt]) - ωzLi*timejt(jt) )
                            AnalAx[ja] = Vxt_[jt]/(ωxLi*m.cos(ωxLi*timejt(jt) + Analφx[ja]))
                            AnalAy[ja] = Vyt_[jt]/(ωyLi*m.cos(ωyLi*timejt(jt) + Analφy[ja]))
                            AnalAz[ja] = Vzt_[jt]/(ωzLi*m.cos(ωzLi*timejt(jt) + Analφz[ja]))   
                        
                        #----------# {{{ Collisions }}}
                        #randcoll = random.uniform(0.,1.)   
                        if (modecoll == 'Cr_M∞'):
                            VLitot_  = sumq3(Vxt_[jt], Vyt_[jt], Vzt_[jt])  
                            Gamma_dt = Γel_vLi(δB_AXIS[jB], VLitot_, n_Gauss(crbath*Xt_[jt],crbath*Yt_[jt],crbath*Zt_[jt], Natom_Cr,sx_Cr,sy_Cr,sz_Cr)) * dt_step  
                        
                        if (modecoll == 'Cr_RealT'):
                            #----# Generate gaussrand Cr velocity
                            if seed is None:         
                                VxCr   = random.gauss(0.0, SV0x_Cr)
                                VyCr   = random.gauss(0.0, SV0_Cr)
                                VzCr   = random.gauss(0.0, SV0_Cr)
                            else:
                                VxCr   = Cr_samples_real[jt,0+ja*3]*SV0x_Cr
                                VyCr   = Cr_samples_real[jt,1+ja*3]*SV0_Cr
                                VzCr   = Cr_samples_real[jt,2+ja*3]*SV0_Cr                            
                            #----# Relative velocity  (before collision)
                            Vrelx  = Vxt_[jt] - VxCr  
                            Vrely  = Vyt_[jt] - VyCr  
                            Vrelz  = Vzt_[jt] - VzCr   
                            Vrelt  = sumq3(Vrelx, Vrely, Vrelz)
                            #----#
                            Gamma_dt = Γel_vrel(δB_AXIS[jB], Vrelt, n_Gauss(crbath*Xt_[jt],crbath*Yt_[jt],crbath*Zt_[jt], Natom_Cr,sx_Cr,sy_Cr,sz_Cr)) * dt_step
                        
                        #-----#
                        Gammadt_avg[jB] = Gammadt_avg[jB] + Gamma_dt
                        if (Gamma_dt > 1.0):
                            FLAG_Gmdt[jB] = FLAG_Gmdt[jB] + 1     # check that Gamma*dt is less than 1
                        
                        if (Gamma_dt > Gammadt_max[jB]):
                            Gammadt_max[jB] = Gamma_dt 
                        

                        #-----# COLLISION STUFF
                        if seed is None:
                            randcoll = random.uniform(0.,1.)
                        else:
                            randcoll = randcoll_all[jt,ja]

                        if (test_jt == jt) and (test_ja == ja):                            
                            print('Cr cls(0)', jt, ja, VxCr, VyCr, VzCr) #, Gammadt_avg[jB], FLAG_Gmdt[jB], Gammadt_max[jB]) 
                            print(Ncoll_atom[ja], randcoll, '<', Gamma_dt, randcoll < Gamma_dt)

                        if (randcoll < Gamma_dt):   #  --> ((Collision happening!))
                            #---# Update counters, calculate MFP stuff
                            Ncoll_atom[ja] = Ncoll_atom[ja] + 1
                            Δt_free        = (1.e3) * (jt - jt_prevcoll) * dt_step     # [ms]
                            Δx_free        = (1.e6) * m.fabs(Xt_[jt] - Xt_[jt_prevcoll])  # [µm]
                            RealΔx_free    = (1.e6) * sum([(m.fabs(Xt_[qx+1] - Xt_[qx])) for qx in range(jt_prevcoll, jt)])   # [µm]   
                            Real_λ_free    = (1.e6) * sum([(sumq3((Xt_[qx+1]-Xt_[qx]), (Yt_[qx+1]-Yt_[qx]), (Zt_[qx+1]-Zt_[qx]))) for qx in range(jt_prevcoll, jt)])  # [µm]     
                            jt_prevcoll    = jt
                            krel_RetTime   = (m_red*Vrelt/h_bar) if (modecoll == 'Cr_RealT') else (m_Li6*VLitot_/h_bar)
                            Retard_Time    = RetardTime(δB_AXIS[jB], krel_RetTime)     # [s], but then saved in [µs]
                            #---# OPENING MOVED OUTSIDE ATOM CYCLE (!!!)
                            #with open(MeanFreePath_det(SimSet,jB), 'a') as mfp:
                            #    print(Δt_free, Δx_free, RealΔx_free, Real_λ_free, (1.e6)*Retard_Time, file=mfp)
                            if save_mfp:
                                print(mfp_fmt_all % (Δt_free, Δx_free, RealΔx_free, Real_λ_free, (1.e6)*Retard_Time), file=mfp)
                            #---# s-wave collision (now good!)
                            #randTheta     = RandNum(0., π)     # ((NB)):  WRONG!!
                            if seed is None:
                                randcosTheta   = random.uniform(-1.,1.)    # = z projection, flat
                                randTheta      = m.acos(randcosTheta)
                                randPhi        = random.uniform(0., 2*π)   # This one instead is fine, indeed is actually good.    
                            else:
                                randcosTheta   = randcosTheta_all[jt][ja]    # = z projection, flat
                                randTheta      = m.acos(randcosTheta)
                                randPhi        = randPhi_all[jt][ja]   # This one instead is fine, indeed is actually good.    
                                
                                if (test_jt == jt) and (test_ja == ja):
                                    print('theta', randcosTheta_all[jt][ja], 'fi', randPhi_all[jt][ja])                        
                                
                            #---#
                            if (modecoll == 'Cr_M∞'):       # Cr has infinite mass, it does not recoil nor transfer/absorb energy from Li, AND IT'S NOT MOVING AT ALL
                                Vxt_[jt]  = VLitot_ * m.sin(randTheta)*m.cos(randPhi)
                                Vyt_[jt]  = VLitot_ * m.sin(randTheta)*m.sin(randPhi)
                                Vzt_[jt]  = VLitot_ * m.cos(randTheta)             
                            
                            if (modecoll == 'Cr_RealT'):    # Cr has FINITE mass and a Temperature, it DOES recoil and CAN transfer/absorb energy from Li 
                                #----# After collision
                                Vrelx_ = Vrelt * m.sin(randTheta)*m.cos(randPhi)
                                Vrely_ = Vrelt * m.sin(randTheta)*m.sin(randPhi)
                                Vrelz_ = Vrelt * m.cos(randTheta)
                                #----# Li exit velocity  (OVERWRITE  Vxt_[jt])
                                Vxt_[jt] = (Vxt_[jt] + MassRat*(VxCr+Vrelx_)) / (1.+MassRat)    # checked, requiring momentum conservation
                                Vyt_[jt] = (Vyt_[jt] + MassRat*(VyCr+Vrely_)) / (1.+MassRat)
                                Vzt_[jt] = (Vzt_[jt] + MassRat*(VzCr+Vrelz_)) / (1.+MassRat)
                                #----#
                            
                            if (test_jt == jt) and (test_ja == ja):
                                print('Li cls(1)', jt, ja, Vxt_[jt], Vyt_[jt], Vzt_[jt])
                            
                            #-------# Re-compute motion parameters (if no RetTime was there)  (this is needed anyway)
                            Analφx[ja] = mod2π( m.atan2((ωxLi*Xt_[jt]), Vxt_[jt]) - ωxLi*timejt(jt) )
                            Analφy[ja] = mod2π( m.atan2((ωyLi*Yt_[jt]), Vyt_[jt]) - ωyLi*timejt(jt) )
                            Analφz[ja] = mod2π( m.atan2((ωzLi*Zt_[jt]), Vzt_[jt]) - ωzLi*timejt(jt) )
                            AnalAx[ja] = Vxt_[jt]/(ωxLi*m.cos(ωxLi*timejt(jt) + Analφx[ja]))
                            AnalAy[ja] = Vyt_[jt]/(ωyLi*m.cos(ωyLi*timejt(jt) + Analφy[ja]))
                            AnalAz[ja] = Vzt_[jt]/(ωzLi*m.cos(ωzLi*timejt(jt) + Analφz[ja]))
                            #-------# Retard time correction on phase (after having computed amplitude)  --> SHIFT ON LI
                            if (RetTimePhase == 1):
                                Analφx[ja] = mod2π( Analφx[ja] - ωxLi*Retard_Time )
                                Analφy[ja] = mod2π( Analφy[ja] - ωyLi*Retard_Time )
                                Analφz[ja] = mod2π( Analφz[ja] - ωzLi*Retard_Time )
                            
                            #-------# Retard time correction on RELATIVE PARTICLE + COM
                            if (RetTimePhase == 2):
                                #----# Cr output velocity (if no ret.time).  NB:  Vxt_[jt] at this stage is ALREADY the Li output velocity
                                VxCr_ = Vxt_[jt] - Vrelx_
                                VyCr_ = Vyt_[jt] - Vrely_
                                VzCr_ = Vzt_[jt] - Vrelz_
                                #----# C.o.M. velocity  (before AND after collision)  [calculated with params AFTER collision IF NO ret time]
                                # Not needed in the end
                                #Vcomx  = (m_Li6*Vxt_[jt] + m_Cr53*VxCr_)/m_LiCr
                                #Vcomy  = (m_Li6*Vyt_[jt] + m_Cr53*VyCr_)/m_LiCr
                                #Vcomz  = (m_Li6*Vzt_[jt] + m_Cr53*VzCr_)/m_LiCr
                                #Vcomt  = sumq3(Vcomx, Vcomy, Vcomz)
                                #----# Li "unperturbed" position/velocity at time (t* + Δt_ret)  [for c.o.m.]
                                xLi__  = AnalAx[ja] * m.sin(ωxLi*(timejt(jt) + Retard_Time) + Analφx[ja])
                                yLi__  = AnalAy[ja] * m.sin(ωyLi*(timejt(jt) + Retard_Time) + Analφy[ja])
                                zLi__  = AnalAz[ja] * m.sin(ωzLi*(timejt(jt) + Retard_Time) + Analφz[ja])
                                vxLi__ = (ωxLi*AnalAx[ja]) * m.cos(ωxLi*(timejt(jt) + Retard_Time) + Analφx[ja])
                                vyLi__ = (ωyLi*AnalAy[ja]) * m.cos(ωyLi*(timejt(jt) + Retard_Time) + Analφy[ja])
                                vzLi__ = (ωzLi*AnalAz[ja]) * m.cos(ωzLi*(timejt(jt) + Retard_Time) + Analφz[ja])
                                #----# Cr motion params (after collision, if no retard time)     [for c.o.m.]
                                Crφx_  = mod2π( m.atan2((ωxCr*Xt_[jt]), VxCr_) - ωxCr*timejt(jt) )
                                Crφy_  = mod2π( m.atan2((ωyCr*Yt_[jt]), VyCr_) - ωyCr*timejt(jt) )
                                Crφz_  = mod2π( m.atan2((ωzCr*Zt_[jt]), VzCr_) - ωzCr*timejt(jt) )
                                CrAx_  = VxCr_/(ωxCr*m.cos(ωxCr*timejt(jt) + Crφx_))
                                CrAy_  = VyCr_/(ωyCr*m.cos(ωyCr*timejt(jt) + Crφy_))
                                CrAz_  = VzCr_/(ωzCr*m.cos(ωzCr*timejt(jt) + Crφz_))
                                #----# Cr "unperturbed" position/velocity at time (t* + Δt_ret)  [for c.o.m.]
                                xCr__  = CrAx_ * m.sin(ωxCr*(timejt(jt) + Retard_Time) + Crφx_)
                                yCr__  = CrAy_ * m.sin(ωyCr*(timejt(jt) + Retard_Time) + Crφy_)
                                zCr__  = CrAz_ * m.sin(ωzCr*(timejt(jt) + Retard_Time) + Crφz_)
                                vxCr__ = (ωxCr*CrAx_) * m.cos(ωxCr*(timejt(jt) + Retard_Time) + Crφx_)
                                vyCr__ = (ωyCr*CrAy_) * m.cos(ωyCr*(timejt(jt) + Retard_Time) + Crφy_)
                                vzCr__ = (ωzCr*CrAz_) * m.cos(ωzCr*(timejt(jt) + Retard_Time) + Crφz_)
                                #----# C.o.M. "unperturbed" position/velocity at time (t* + Δt_ret)
                                xcom__ = WeightAvg2(xLi__,m_Li6, xCr__,m_Cr53)
                                ycom__ = WeightAvg2(yLi__,m_Li6, yCr__,m_Cr53)
                                zcom__ = WeightAvg2(zLi__,m_Li6, zCr__,m_Cr53)
                                vxcom__= WeightAvg2(vxLi__,m_Li6, vxCr__,m_Cr53)
                                vycom__= WeightAvg2(vyLi__,m_Li6, vyCr__,m_Cr53)
                                vzcom__= WeightAvg2(vzLi__,m_Li6, vzCr__,m_Cr53)
                                #----# Fix passage at time (t* + Δt_ret) in new Li trajectory
                                xListar  = xcom__  # relative distance = 0
                                yListar  = ycom__
                                zListar  = zcom__
                                VxListar = vxcom__ + MassRat/(1.+MassRat) * Vrelx_
                                VyListar = vycom__ + MassRat/(1.+MassRat) * Vrely_
                                VzListar = vzcom__ + MassRat/(1.+MassRat) * Vrelz_
                                #----# New Motion params (final ones)
                                Analφx[ja] = mod2π( m.atan2((ωxLi*xListar), VxListar) - ωxLi*(timejt(jt)+Retard_Time) )
                                Analφy[ja] = mod2π( m.atan2((ωyLi*yListar), VyListar) - ωyLi*(timejt(jt)+Retard_Time) )
                                Analφz[ja] = mod2π( m.atan2((ωzLi*zListar), VzListar) - ωzLi*(timejt(jt)+Retard_Time) )
                                AnalAx[ja] = VxListar/(ωxLi*m.cos(ωxLi*(timejt(jt)+Retard_Time) + Analφx[ja]))
                                AnalAy[ja] = VyListar/(ωyLi*m.cos(ωyLi*(timejt(jt)+Retard_Time) + Analφy[ja]))
                                AnalAz[ja] = VzListar/(ωzLi*m.cos(ωzLi*(timejt(jt)+Retard_Time) + Analφz[ja]))
                                #----#

                            if (test_jt == jt) and (test_ja == ja):
                                print('Li cls(2)', jt, ja, Analφx[ja], Analφy[ja], Analφz[ja], AnalAx[ja], AnalAy[ja], AnalAz[ja])
                        
                            #-------#
                        
                        #---------#   
                
                    #-----------------------# [[[ END Trajectory cycle]]]
                    #-----------------------#
                    #-----------------------#
                    #-----# Calculate (EnergyCons)   ---> print moved outside ATOM cycle, before END DET cycle. E_tot arrays are NOT overwritten
                    Etot_fin[ja] = 0.5*m_Li6*sumq3(Vxt_[N_tsteps-1], Vyt_[N_tsteps-1], Vzt_[N_tsteps-1])**2  +  \
                                0.5*m_Li6*sumq3((ωxLi*Xt_[N_tsteps-1]), (ωyLi*Yt_[N_tsteps-1]), (ωzLi*Zt_[N_tsteps-1]))**2
                    #with open(EnergyCons_det(SimSet,jB), 'a') as energycons:
                    #    print(δB_AXIS[jB], ja+1, Etot_in[ja], Etot_fin[ja], Ncoll_atom[ja], file=energycons)
                    
                    #-----# Print (Coords05)  --> NB:  Xt_,... arrays are OVERWRITTEN!  but opening can be moved
                    #with open(Coords05_det(SimSet,jB), 'a') as coords05:
                    for j05 in range(N_05+1):
                        jmult05_ = jCorrespTo_time(t05_from_j05(j05))
                        print(coords_fmt_all % (δB_AXIS[jB], ja+1, t05_from_j05(j05), \
                            (1.e6*Xt_[jmult05_]), (1.e6*Yt_[jmult05_]), (1.e6*Zt_[jmult05_]), \
                            (1.e3*Vxt_[jmult05_]), (1.e3*Vyt_[jmult05_]), (1.e3*Vzt_[jmult05_])), file=coords05)
                        
                    #----# add spaces if un-comment with open
                    #-----# λ_free for zero collisions (& last point anyway)
                    if (jt_prevcoll != (N_tsteps-1) and SaveLastPoint):
                        Δt_free        = (1.e3) * ((N_tsteps-1) - jt_prevcoll) * dt_step
                        Δx_free        = (1.e6) * m.fabs(Xt_[N_tsteps-1] - Xt_[jt_prevcoll])
                        RealΔx_free    = (1.e6) * sum([(m.fabs(Xt_[qx+1] - Xt_[qx])) for qx in range(jt_prevcoll, N_tsteps-1)])
                        Real_λ_free    = (1.e6) * sum([(sumq3((Xt_[qx+1]-Xt_[qx]), (Yt_[qx+1]-Yt_[qx]), (Zt_[qx+1]-Zt_[qx]))) for qx in range(jt_prevcoll, N_tsteps-1)])
                        Retard_Time    = 0.0
                        #----# Commented out for a while
                        #with open(MeanFreePath_det(SimSet,jB), 'a') as mfp:
                        #    print(Δt_free, Δx_free, RealΔx_free, Real_λ_free, Retard_Time, file=mfp)  
                        
                        #----#
                
                #-----#
            
            #--------------------------------#  (((END:   Atoms/Histories cycle)))
        
    
    if False:
        #--------# Print EnergyCons  (NEW position!)
        with open(EnergyCons_det(SimSet,jB), 'a') as energycons:
            for ja in range(N_atoms):  
                print(δB_AXIS[jB], ja+1, Etot_in[ja], Etot_fin[ja], Ncoll_atom[ja], file=energycons)
        
    
    #--------#
    #--------#
    Gammadt_avg[jB] = Gammadt_avg[jB]/N_atoms/(N_tsteps-1)
    #--------# Save avg number of collisions at this detuning
    Ncoll_avg[jB] = sum([(Ncoll_atom[nn]) for nn in range(N_atoms)]) / N_atoms
    #Ncoll_avg_e[jB]  = STATS_stddev  # maybe STATS_mean_err is better in this case?
    #Ncoll_avg_e2[jB] = STATS_mean_err
    #--------#

#--------------------------------------------------#  [[[END Detuning cycle]]]
#------------------------------------#
#----------------------#
#--------#

if False:
    #--------# Save avg Ncoll [for this instance]
    with open(AvgNcoll_inst(SimSet,SimInstn), 'w') as avgncoll:
        print("Det(mG)\t Ncoll_avg \t Ncoll_err \t Ncoll_meanerr", file=avgncoll)
        for jB in range(N_ptsδB):
            print(δB_AXIS[jB], "\t", Ncoll_avg[jB], file=avgncoll)

    #--------# Save FLAGScoll [for this instance]
    with open(FLAGScoll_inst(SimSet,SimInstn), 'w') as flagscoll:
        print("Det(mG)\t NFLAGS_tot \t Gammadt_max \t Gammadt_avg \t dt_step (µs)", file=flagscoll)
        for jB in range(N_ptsδB):
            print( δB_AXIS[jB], "\t", FLAG_Gmdt[jB], "\t", Gammadt_max[jB], "\t",  \
                   Gammadt_avg[jB], "\t", 1.e6*(DTSTEP[IsCloseToRes_dt(δB_AXIS[jB])]), file=flagscoll)

#--------#

print('calculation time %.3fs\n' % (time.time()-t_start))

