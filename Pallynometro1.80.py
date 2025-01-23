#-------------------------------------------#
#--------#     Pinball  Machine    #--------#
#--------#      ...on Python!      #--------#
#-------------------------------------------#
#--------#
#  load SpinDiff_anal."Pallinometri/PinballMachines_SettingsMathFunctions_1.00.gp"
#--------#
import math as m
import numpy
import random
import codecs
#-----#
import sys
sys.stdout.reconfigure(encoding='utf-8')
#sys.path.insert(0, 'C:/Users/stefa/Desktop/PhD/Code/python/')
#from Pallynometro_Header import *
#-----#


#----------------------------------------------# [Numerical Simulator]   
#------------------------------#
#------#  [Sim Params]  #------#  
#------------------------------#
#--------# Notes
# 
# 
# 
#--------# "Parallelization" settings
SimSet   = '204'
SimInstn = 6       # instance
#--------# Atomic sample numbers: ((Li))
TGaussLi = 0.035e-6     # T_x
TGauzzLi = 0.43e-6     # T_z, T_y
SX0_Li   = 37.0e-6    
SZ0_Li   = 5.0e-6
SY0_Li   = 6.5e-6     # taking into account effect of trap freqs
SV0_Li   = sqrt(k_B*TGaussLi/m_Li6)  # sigma v_x
SV0z_Li  = sqrt(k_B*TGauzzLi/m_Li6)  # sigma v_z, v_y
ωxLi     = (2*π) * 16.7   # x = axial (along ODT axis)         # NB: the final one w/o cross
ωyLi     = (2*π) * 530.   # y = radial (don't see, along img)  # NB: the final one w/o cross
ωzLi     = (2*π) * 680.   # z = vertical (along gravity)       # NB: the final one w/o cross (irrelevant here)
#--------# Atomic sample numbers: [[Cr]]
Natom_Cr = 120.e3
sx_Cr    = 60.e-6
sz_Cr    = 6.0e-6
sy_Cr    = 7.5e-6
TGaussCr = 0.14e-6  # NB: HERE for y and z
TGauxxCr = 0.18e-6  # NB: for x
SV0_Cr   = sqrt(k_B*TGaussCr/m_Cr53)
SV0x_Cr  = sqrt(k_B*TGauxxCr/m_Cr53)
MassRat  = m_Cr53/m_Li6 
#m_red   = m_Li6 * MassRat/(1.+MassRat) 
m_red    = (m_Cr53*m_Li6)/(m_Cr53+m_Li6)
m_LiCr   = (m_Li6 + m_Cr53)
ωxCr     = (2*π) * 13.8   # x = axial (along ODT axis)         # NB: the final one w/o cross
ωyCr     = (2*π) *  88.   # y = radial (don't see, along img)  # NB: the final one w/o cross
ωzCr     = (2*π) * 112.   # z = vertical (along gravity)       # NB: the final one w/o cross (irrelevant here)
#--------# Simulation settings
N_atoms  = 10000    # NB: not real atom number, but number of simulated atoms/trajectories/histories
t_final  = 40.e-3   # [40 ms]
modeEvol = 'analytic'   #  'analytic', 'numericMF'
modecoll = 'Cr_RealT'   #  'Cr_M∞', 'Cr_RealT'
modebath = 'Cr_Gauss'   #  'Cr_Gauss', 'Cr_homog'
#--------# Time steps
DTSTEP   = [100.e-6, 50.e-6, 25.e-6, 10.e-6,  5.e-6,  2.e-6,  1.e-6,  0.25e-6]      
def IsCloseToRes_dt(dB):
    if (dB >= -1. and dB <= 6.):   # was up to 4.
        return 5
    elif (dB>= -3. and dB<=10.):
        return 4
    elif (dB>=-10. and dB<=40.):
        return 3
    elif (dB>=-20. and dB<=60.):
        return 2
    else:
        return 1    # 100 is not used in this case

#---#
#DTSTEP   = [ 100.e-6, 50.e-6, 25.e-6, 10.e-6,  5.e-6,  2.e-6,  1.e-6,  0.50E-6,  0.25e-6,  0.10e-6 ]      
#def IsCloseToRes_dt(dB):
#    if (dB == -1. or dB == 1.):
#        return 9
#    else:
#        return 8

#--------# Simulation settings 2.0
EnableColl    = 1                  #  0, 1: disables/enables collisions
VRELcoll      = 'RandVCr'          #  'RandVCr', 'SigmaVCr':  how v_rel is computed for collision prob.
SameCrcoll    = 0                  #  0, 1:  for correlated collisions with same Cr atom
SaveLastPoint = 0                  #  just for saving info in mfp, keep it 0
RetTimePhase  = 0                  #  0 = NO ret. time, 1 = Cr fixed scatt. center, 2 = Cr moving
BlockRetColls = 0
MeanFieldDima = 0
PreparationRF = 0   # may have some problems 
BFieldDrift   = 0   # half-span BDrift   [mG]    pos= BCS-->BEC
#--------# Preparation Params
t_prep   = 1.0e-3   # duration of preparation: from beginning of Cr pulse, to cross opening
tp_RFCr  = 0.9e-3   # duration Cr RF π-pulse
ΩRabi_Cr = π/tp_RFCr
ωxLi_pre = (2*π) * 60.    # careful with this one!
ωyLi_pre = (2*π) * 535.
ωzLi_pre = (2*π) * 680.
#--------# Further   (set to zero if you want to remove them)
blevgrad = 0 * 1.5 * 1.e5     # [G/cm]   * 1.e5 = [mG/m]
Bcurv_x  = 0 * 12. * 1.e7     # [G/cm^2] * 1.e7 = [mG/m^2]
#--------#

#--------# Pre-calculation of consts  (again, if artificially change MCr)
mred_hbar = m_red/h_bar
mred_mLi6 = m_red/m_Li6
mLi6mred  = m_Li6*m_red
#--------# Sampling every 0.5ms: functions
N_05 = int(t_final/0.5e-3)    # without +1 if j05 starts from 0
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
    δB_AXIS = [-100., -95., -90., -85., -80., -75., -70., -65., -60.,    \
                -55., -50., -45., -40., -36., -32., -30., -28., -26.,    \
                -24., -22., -20., -18., -16., -14., -12.,                \
                -10.,  -9.,  -8.,  -7.,  -6.,  -5.,  -4.,  -3.,  -2.,  -1.,  \
                  1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,  \
                 12.,  14.,  16.,  18.,  20.,  22.,  24.,  26.,  28.,  30.,  \
                 32.,  34.,  36.,  38.,  40.,  45.,  50.,  55.,  60.,  65.,  \
                 70.,  75.,  80.,  85.,  90.,  95.,  100.  ]

#-------#
if (SimInstn == 1):      # (1), 65 comp_t_units
    δB_AXIS = [-100., -95., -90., -85., -80., -75., -70., -65., -60.,    \
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
if (SimInstn == 21):      # (21), 1 comp_t_units
    δB_AXIS = [ -100. ]
elif (SimInstn == 22):    # (22), 1 comp_t_units
    δB_AXIS = [  -10. ]
elif (SimInstn == 23):    # (23), 1 comp_t_units
    δB_AXIS = [   -1. ]
elif (SimInstn == 24):    # (24), 1 comp_t_units
    δB_AXIS = [    1. ]   
elif (SimInstn == 25):    # (25), 1 comp_t_units
    δB_AXIS = [   10. ]
elif (SimInstn == 26):    # (26), 1comp_t_units
    δB_AXIS = [  100. ]  

#-------#
if (SimInstn == 98):      # 
    δB_AXIS = [-100., -80., -60., -40., -20.,  20., 40., 60., 80., 100.]

if (SimInstn == 99):      # 
    δB_AXIS = [-10000., -1000., -500., -200., -100., 100., 200., 500., 1000., 10000. ]

#-------#
if (SimInstn == 77):      # 
    δB_AXIS = [ -2., -1., 1., 2. ]

#-------#
if (SimInstn == 7):      # 
    δB_AXIS = [ 1., 2., 3., 4., 5., 6. ]

#-------#
N_ptsδB = len(δB_AXIS)   
#-------# Arrays to save info about collisions           
Ncoll_avg = [0.0] * N_ptsδB  # --> add also errors!!
FLAG_Gmdt = [0] * N_ptsδB
Gammadt_max = [0.0] * N_ptsδB
Gammadt_avg = [0.0] * N_ptsδB

PreNcoll_avg = [0.0] * N_ptsδB  # during preparation RF
PreFLAG_Gmdt = [0] * N_ptsδB
PreGammadt_max = [0.0] * N_ptsδB
PreGammadt_avg = [0.0] * N_ptsδB
#-------#

#-------# Arrays to save info about propagation:
#if (modeEvol == "numericMF"):
#    xdot_avg  = [0.0] * N_ptsδB;  ydot_avg  = [0.0] * N_ptsδB;  zdot_avg  = [0.0] * N_ptsδB   # \dot{x} * dt_step, and similar
#    vxdot_avg = [0.0] * N_ptsδB;  vydot_avg = [0.0] * N_ptsδB;  vzdot_avg = [0.0] * N_ptsδB

#------------------------------------#



#--------#
#----------------------#
#------------------------------------#
#--------------------------------------------------#  [[[BEGIN Detuning cycle]]]
for jB in range(N_ptsδB):
    #-----#
    if (BFieldDrift != 0):
        Central_δB  = δB_AXIS[jB]
    
    #-------# EnergyCons & coords (now GLOBAL) 
    with open(EnergyCons_det(SimSet,jB), 'w') as energycons:
        print("Det  (mG) \t AtomIndx \t Etot_in (J) \t Etot_fin (J) \t N_coll \t Epot_in \t Ekin_in \t Epot_fin \t Ekin_fin", file=energycons)
    
    with open(Coords05_det(SimSet,jB), 'w') as coords05:
        print("Det  (mG) \t AtomIndx \t t (ms) \t x (µm) \t y (µm) \t z (µm) \t v_x (µm/ms) \t v_y (µm/ms) \t v_z (µm/ms)", file=coords05)
    
    with open(MeanFreePath_det(SimSet,jB), 'w') as mfp:
        print("Dt_free(ms) \t Dx_free(µm) \t RealDx_free(µm) \t Real_l_free(µm) \t RetardTime(µs)", file=mfp)
    
    #--------# Generate initial distributions (re-do for every detuning)   --> can generate all at once?
    X_in  = [None] * N_atoms;  Y_in = [None] * N_atoms;  Z_in = [None] * N_atoms; 
    Vx_in = [None] * N_atoms; Vy_in = [None] * N_atoms; Vz_in = [None] * N_atoms; 
    for ja in range(N_atoms):
        X_in[ja]  = random.gauss(0.0, SX0_Li)    # center = 0
        Y_in[ja]  = random.gauss(0.0, SY0_Li)  
        Z_in[ja]  = random.gauss(0.0, SZ0_Li)
        Vx_in[ja] = random.gauss(0.0, SV0_Li) 
        Vy_in[ja] = random.gauss(0.0, SV0z_Li)
        Vz_in[ja] = random.gauss(0.0, SV0z_Li)
    
    #--------# Declare some arrays (re-initialized at every detuning)
    Ncoll_atom    = [0] * N_atoms
    PreNcoll_atom = [0] * N_atoms
    Etot_in    = [None] * N_atoms; Etot_fin = [None] * N_atoms   # just mechanical (kin+pot), no MF
    Epot_in    = [None] * N_atoms; Epot_fin = [None] * N_atoms
    Ekin_in    = [None] * N_atoms; Ekin_fin = [None] * N_atoms
    if (modeEvol == "analytic"):
        AnalAx = [None] * N_atoms;     AnalAy = [None] * N_atoms;  AnalAz = [None] * N_atoms   # consider not having them as arrays/lists
        Analφx = [None] * N_atoms;     Analφy = [None] * N_atoms;  Analφz = [None] * N_atoms   # but as simple variables
    
    if (PreparationRF == 1):
        PreAnalAx = [None] * N_atoms;     PreAnalAy = [None] * N_atoms;  PreAnalAz = [None] * N_atoms  
        PreAnalφx = [None] * N_atoms;     PreAnalφy = [None] * N_atoms;  PreAnalφz = [None] * N_atoms
    
    #--------# Optimize dt_step  
    dt_step  = DTSTEP[IsCloseToRes_dt(δB_AXIS[jB])]
    N_tsteps = int(t_final/dt_step + 1.e-5) + 1  # including first step @ t=0
    #--------# 
    
    #---------------------------------------# [[[ Preparation stage with Cr RF ]]]
    #---------------------------------------# 
    if (PreparationRF == 1):
        #----# N t_steps for preparation stage  (from -1 ms to 0)
        N_presteps  = int(t_prep/dt_step + 1.e-5) + 1    # including first step @ t=0  (t=-1ms actually here)
        #----#
        #-------------------------------# (((Atom cycle Preparation)))
        for ja in range(N_atoms):  
            #-----# Trajectory arrays for prep stage
            Xt_pre  = [None] * N_presteps;  Xt_pre[0]  = X_in[ja]
            Yt_pre  = [None] * N_presteps;  Yt_pre[0]  = Y_in[ja]
            Zt_pre  = [None] * N_presteps;  Zt_pre[0]  = Z_in[ja]
            Vxt_pre = [None] * N_presteps;  Vxt_pre[0] = Vx_in[ja]
            Vyt_pre = [None] * N_presteps;  Vyt_pre[0] = Vy_in[ja]
            Vzt_pre = [None] * N_presteps;  Vzt_pre[0] = Vz_in[ja]
            #-----# Initialize Pre-A and Pre-φ arrays for analmode (t=-1)
            if (modeEvol == "analytic"):
                PreAnalφx[ja] = m.atan2((ωxLi_pre*X_in[ja]), Vx_in[ja]) % twoπ     
                PreAnalφy[ja] = m.atan2((ωyLi_pre*Y_in[ja]), Vy_in[ja]) % twoπ     
                PreAnalφz[ja] = m.atan2((ωzLi_pre*Z_in[ja]), Vz_in[ja]) % twoπ     
                PreAnalAx[ja] = Vx_in[ja]/(ωxLi_pre*m.cos(PreAnalφx[ja]))  
                PreAnalAy[ja] = Vy_in[ja]/(ωyLi_pre*m.cos(PreAnalφy[ja]))  
                PreAnalAz[ja] = Vz_in[ja]/(ωzLi_pre*m.cos(PreAnalφz[ja]))
            
            #---------------------------# [[Preparation trajectory cycle]]
            for jjtt in range(1, N_presteps): 
                timejjtt_ = jjtt * dt_step
                #----# Variable Cr atom number (Rabi osci, stopped at 0.9 ms)
                if (timejjtt_ < tp_RFCr):
                    PreNatom_Cr = Natom_Cr * (sin(0.5*ΩRabi_Cr*timejjtt_))**2
                else:
                    PreNatom_Cr = Natom_Cr
                #----------# Propagation for one dt_step  (analytic harmonic oscillator)
                if (modeEvol == "analytic"):
                    Xt_pre[jjtt]  = PreAnalAx[ja] * m.sin(ωxLi_pre*timejjtt_ + PreAnalφx[ja])
                    Yt_pre[jjtt]  = PreAnalAy[ja] * m.sin(ωyLi_pre*timejjtt_ + PreAnalφy[ja])
                    Zt_pre[jjtt]  = PreAnalAz[ja] * m.sin(ωzLi_pre*timejjtt_ + PreAnalφz[ja])
                    Vxt_pre[jjtt] = (ωxLi_pre*PreAnalAx[ja]) * m.cos(ωxLi_pre*timejjtt_ + PreAnalφx[ja])
                    Vyt_pre[jjtt] = (ωyLi_pre*PreAnalAy[ja]) * m.cos(ωyLi_pre*timejjtt_ + PreAnalφy[ja])
                    Vzt_pre[jjtt] = (ωzLi_pre*PreAnalAz[ja]) * m.cos(ωzLi_pre*timejjtt_ + PreAnalφz[ja])
                
                #----------# {{{ Collisions }}}
                randcoll = random.uniform(0.,1.)
                if (modecoll == 'Cr_RealT'):
                    #----# Generate gaussrand Cr velocity           
                    VxCr   = random.gauss(0.0, SV0x_Cr)
                    VyCr   = random.gauss(0.0, SV0_Cr)
                    VzCr   = random.gauss(0.0, SV0_Cr)
                    #----# Relative velocity  (before collision)
                    Vrelx  = Vxt_pre[jjtt] - VxCr  
                    Vrely  = Vyt_pre[jjtt] - VyCr  
                    Vrelz  = Vzt_pre[jjtt] - VzCr   
                    Vrelt  = m.sqrt(Vrelx*Vrelx + Vrely*Vrely + Vrelz*Vrelz)
                    #----# NB: no crbath here (but can add), but VARIABLE PreNatom_Cr (!)
                    Gamma_dt = Γel_vrel(DetmG(δB_AXIS[jB],Xt_pre[jjtt],Zt_pre[jjtt]), Vrelt, \
                               n_Gauss(Xt_pre[jjtt],Yt_pre[jjtt],Zt_pre[jjtt], PreNatom_Cr,sx_Cr,sy_Cr,sz_Cr)) * dt_step
                
                #-----#
                PreGammadt_avg[jB] = PreGammadt_avg[jB] + Gamma_dt
                if (Gamma_dt > 1.0):
                    PreFLAG_Gmdt[jB] = PreFLAG_Gmdt[jB] + 1     # check that Gamma*dt is less than 1
                
                if (Gamma_dt > PreGammadt_max[jB]):
                    PreGammadt_max[jB] = Gamma_dt 
                
                #----------------------------------#   --> ((Collision happening!))
                if (randcoll < EnableColl*Gamma_dt):     
                    #---# Update counters, calculate MFP stuff
                    PreNcoll_atom[ja] = PreNcoll_atom[ja] + 1
                    #Δt_free        = (1.e3) * (jt - jt_prevcoll) * dt_step     # [ms]
                    #Δx_free        = (1.e6) * m.fabs(Xt_[jt] - Xt_[jt_prevcoll])  # [µm]
                    #RealΔx_free    = (1.e6) * sum([(m.fabs(Xt_[qx+1] - Xt_[qx])) for qx in range(jt_prevcoll, jt)])   # [µm]   
                    #Real_λ_free    = (1.e6) * sum([(sumq3((Xt_[qx+1]-Xt_[qx]), (Yt_[qx+1]-Yt_[qx]), (Zt_[qx+1]-Zt_[qx]))) for qx in range(jt_prevcoll, jt)])  # [µm]     
                    #jt_prevcoll    = jt
                    #krel_RetTime   = (m_red*Vrelt/h_bar) if (modecoll == 'Cr_RealT') else (m_Li6*VLitot_/h_bar)
                    #Retard_Time    = RetardTime(δB_AXIS[jB], krel_RetTime)     # [s], but then saved in [µs]
                    #---# s-wave collision (now good!)
                    randcosTheta   = random.uniform(-1.,1.)    # = z projection, flat
                    randTheta      = m.acos(randcosTheta)
                    randPhi        = random.uniform(0., 2*π)   # This one instead is fine, indeed is actually good.    
                    #---#
                    if (modecoll == 'Cr_M∞'):       # Cr has infinite mass, it does not recoil nor transfer/absorb energy from Li, AND IT'S NOT MOVING AT ALL
                        Vxt_pre[jjtt]  = VLitot_ * m.sin(randTheta)*m.cos(randPhi)
                        Vyt_pre[jjtt]  = VLitot_ * m.sin(randTheta)*m.sin(randPhi)
                        Vzt_pre[jjtt]  = VLitot_ * m.cos(randTheta)             
                    elif (modecoll == 'Cr_RealT'):  # Cr has FINITE mass and a Temperature, it DOES recoil and CAN transfer/absorb energy from Li 
                        #----# After collision
                        Vrelx_ = Vrelt * m.sin(randTheta)*m.cos(randPhi)
                        Vrely_ = Vrelt * m.sin(randTheta)*m.sin(randPhi)
                        Vrelz_ = Vrelt * m.cos(randTheta)
                        #----# Li exit velocity  (OVERWRITE  Vxt_pre[jjtt])     ((NB!))
                        Vxt_pre[jjtt] = (Vxt_pre[jjtt] + MassRat*(VxCr+Vrelx_)) / (1.+MassRat)    # checked, requiring momentum conservation
                        Vyt_pre[jjtt] = (Vyt_pre[jjtt] + MassRat*(VyCr+Vrely_)) / (1.+MassRat)
                        Vzt_pre[jjtt] = (Vzt_pre[jjtt] + MassRat*(VzCr+Vrelz_)) / (1.+MassRat)
                        #----#
                    
                    #-------# Re-compute motion parameters (if no RetTime was there)  (this is needed anyway)
                    if (modeEvol == "analytic"):
                        PreAnalφx[ja] = ( m.atan2((ωxLi_pre*Xt_pre[jjtt]), Vxt_pre[jjtt]) - ωxLi_pre*timejjtt_ ) % twoπ
                        PreAnalφy[ja] = ( m.atan2((ωyLi_pre*Yt_pre[jjtt]), Vyt_pre[jjtt]) - ωyLi_pre*timejjtt_ ) % twoπ
                        PreAnalφz[ja] = ( m.atan2((ωzLi_pre*Zt_pre[jjtt]), Vzt_pre[jjtt]) - ωzLi_pre*timejjtt_ ) % twoπ
                        PreAnalAx[ja] = Vxt_pre[jjtt]/(ωxLi_pre*m.cos(ωxLi_pre*timejjtt_ + PreAnalφx[ja]))
                        PreAnalAy[ja] = Vyt_pre[jjtt]/(ωyLi_pre*m.cos(ωyLi_pre*timejjtt_ + PreAnalφy[ja]))
                        PreAnalAz[ja] = Vzt_pre[jjtt]/(ωzLi_pre*m.cos(ωzLi_pre*timejjtt_ + PreAnalφz[ja]))
                    
                    #-------#
                
                #----------------------------------#  ((END collision))
            #---------------------------# [[END:  Preparation trajectory cycle]]
            
            #---------# {{{OUTCOME PREP_RF}}}: Re-assign X_in, ...
            X_in[ja]  = Xt_pre[N_presteps-1]
            Y_in[ja]  = Yt_pre[N_presteps-1]
            Z_in[ja]  = Zt_pre[N_presteps-1]
            Vx_in[ja] = Vxt_pre[N_presteps-1]
            Vy_in[ja] = Vyt_pre[N_presteps-1]
            Vz_in[ja] = Vzt_pre[N_presteps-1]
            #---------#
        
        #-------------------------------# (((END:  Atom cycle Preparation)))
    else:
        N_presteps  = 0
    
    #---------------------------------------# [[[ END:  Preparation stage with Cr RF ]]]
    #---------------------------------------#
    
    #---------------------------------------# {{{BEGIN:  REAL SIM}}}
    #--------# NB: NEW POSITION OPEN MFP   (!!!)
    with open(MeanFreePath_det(SimSet,jB), 'a') as mfp:
        with open(Coords05_det(SimSet,jB), 'a') as coords05:
            #--------------------------------#  (((BEGIN:   Atoms/Histories cycle)))
            for ja in range(N_atoms):  
                EnableColl  = 1     # NB: comment this, beside putting it =0 at the beginning, to disable coll
                jt_prevcoll = 0     # python arrays start from 0  
                RecentColl  = 0      
                Epot_in[ja] = 0.5*m_Li6*sumq3((ωxLi*X_in[ja]), (ωyLi*Y_in[ja]), (ωzLi*Z_in[ja]))**2
                Ekin_in[ja] = 0.5*m_Li6*(Vx_in[ja]*Vx_in[ja] + Vy_in[ja]*Vy_in[ja] + Vz_in[ja]*Vz_in[ja])
                Etot_in[ja] = Epot_in[ja] + Ekin_in[ja]
                #-----# Declare & initialize trajectory arrays 
                Xt_  = [None] * N_tsteps;  Xt_[0]  = X_in[ja]
                Yt_  = [None] * N_tsteps;  Yt_[0]  = Y_in[ja]
                Zt_  = [None] * N_tsteps;  Zt_[0]  = Z_in[ja]
                Vxt_ = [None] * N_tsteps;  Vxt_[0] = Vx_in[ja]
                Vyt_ = [None] * N_tsteps;  Vyt_[0] = Vy_in[ja]
                Vzt_ = [None] * N_tsteps;  Vzt_[0] = Vz_in[ja]
                #-----# Initialize A and φ arrays for analmode (t=0)
                if (modeEvol == "analytic"):
                    Analφx[ja] = m.atan2((ωxLi*X_in[ja]), Vx_in[ja]) % twoπ      # atan2 is def btw -π and +π
                    Analφy[ja] = m.atan2((ωyLi*Y_in[ja]), Vy_in[ja]) % twoπ      # trick: keep atan2 here, and mod2pi only after collisions
                    Analφz[ja] = m.atan2((ωzLi*Z_in[ja]), Vz_in[ja]) % twoπ      #        then you get a timescale for dephasing
                    AnalAx[ja] = Vx_in[ja]/(ωxLi*m.cos(Analφx[ja]))              # a posteriori: "trick" not really relevant maybe?
                    AnalAy[ja] = Vy_in[ja]/(ωyLi*m.cos(Analφy[ja]))  
                    AnalAz[ja] = Vz_in[ja]/(ωzLi*m.cos(Analφz[ja]))  
                
                #-------------------------------------# [[[ BEGIN Trajectory cycle]]]  (dt_steps)
                for jt in range(1, N_tsteps): 
                    timejt_ = jt * dt_step
                    #----------# External dynamics (by hand)
                    if (BFieldDrift != 0):
                        δB_AXIS[jB] = (Central_δB+BFieldDrift) - (2*BFieldDrift/t_final) * timejt_
                        if (m.fabs(δB_AXIS[jB]) < 0.5):
                            δB_AXIS[jB] = 0.5 * numpy.sign(δB_AXIS[jB] + 1.71e-3)   # I have no idea why I put this 1.71e-3 here...
                        
                    
                    #----------# Propagation for one dt_step  (analytic harmonic oscillator OR numerical w mean-field)
                    if   (modeEvol == "analytic"):
                        Xt_[jt]  = AnalAx[ja] * m.sin(ωxLi*timejt_ + Analφx[ja])
                        Yt_[jt]  = AnalAy[ja] * m.sin(ωyLi*timejt_ + Analφy[ja])
                        Zt_[jt]  = AnalAz[ja] * m.sin(ωzLi*timejt_ + Analφz[ja])
                        Vxt_[jt] = ωxLi*AnalAx[ja] * m.cos(ωxLi*timejt_ + Analφx[ja])
                        Vyt_[jt] = ωyLi*AnalAy[ja] * m.cos(ωyLi*timejt_ + Analφy[ja])
                        Vzt_[jt] = ωzLi*AnalAz[ja] * m.cos(ωzLi*timejt_ + Analφz[ja])
                        if (SameCrcoll == 1 and RecentColl != 0):
                            XCrClose  = AnalAx_Cr * m.sin(ωxCr*timejt_ + Analφx_Cr)
                            YCrClose  = AnalAy_Cr * m.sin(ωyCr*timejt_ + Analφy_Cr)
                            ZCrClose  = AnalAz_Cr * m.sin(ωzCr*timejt_ + Analφz_Cr)
                            VXCrClose = ωxCr*AnalAx_Cr * m.cos(ωxCr*timejt_ + Analφx_Cr)
                            VYCrClose = ωyCr*AnalAy_Cr * m.cos(ωyCr*timejt_ + Analφy_Cr)
                            VZCrClose = ωzCr*AnalAz_Cr * m.cos(ωzCr*timejt_ + Analφz_Cr)
                            #-----# Relative distance after 1 dt_step
                            RelDist_x = Xt_[jt] - XCrClose
                            RelDist_y = Yt_[jt] - YCrClose
                            RelDist_z = Zt_[jt] - ZCrClose
                            RelDist_  = m.sqrt(RelDist_x*RelDist_x + RelDist_y*RelDist_y + RelDist_z*RelDist_z)
                            #-----# New relative velocity
                            NewVrel_x = Vxt_[jt] - VXCrClose
                            NewVrel_y = Vyt_[jt] - VYCrClose
                            NewVrel_z = Vzt_[jt] - VZCrClose
                            NewVrel_  = m.sqrt(NewVrel_x*NewVrel_x + NewVrel_y*NewVrel_y + NewVrel_z*NewVrel_z)
                            #-----# Effective hard-sphere radius  
                            EffRadius = m.sqrt( σ_el(mred_hbar*NewVrel_, a_Bohr*as_1461G(DetmG(δB_AXIS[jB],Xt_[jt],Zt_[jt])), a_Bohr*Rs_1461G(DetmG(δB_AXIS[jB],Xt_[jt],Zt_[jt]))) / fourπ )
                            if (RelDist_ > EffRadius):
                                RecentColl = 0
                            
                            #-----#
                    elif (modeEvol == "numericMF"):   
                        #------# Randomly-picked Cr velocity   (used only in mode RandV*, manually set by Header now)
                        VxCrMF = random.gauss(0.0, SV0x_Cr)
                        VyCrMF = random.gauss(0.0, SV0_Cr)
                        VzCrMF = random.gauss(0.0, SV0_Cr)
                        #------# Linear increments/K1 coefficients (Euler's method).  NB: WITH dt_step also here!!
                        DetmG1 = DetmG(δB_AXIS[jB],Xt_[jt-1],Zt_[jt-1])  # calculate once to save time
                        K1_x   = ( Vxt_[jt-1] + MFHam_x(Xt_[jt-1],Yt_[jt-1],Zt_[jt-1], Vxt_[jt-1],Vyt_[jt-1],Vzt_[jt-1], DetmG1 ) ) * dt_step
                        K1_y   = ( Vyt_[jt-1] + MFHam_y(Xt_[jt-1],Yt_[jt-1],Zt_[jt-1], Vxt_[jt-1],Vyt_[jt-1],Vzt_[jt-1], DetmG1 ) ) * dt_step 
                        K1_z   = ( Vzt_[jt-1] + MFHam_z(Xt_[jt-1],Yt_[jt-1],Zt_[jt-1], Vxt_[jt-1],Vyt_[jt-1],Vzt_[jt-1], DetmG1 ) ) * dt_step 
                        K1_vx  = ( -ωxLi**2 * Xt_[jt-1] + MFHam_vx(Xt_[jt-1],Yt_[jt-1],Zt_[jt-1], Vxt_[jt-1],Vyt_[jt-1],Vzt_[jt-1], DetmG1 ) ) * dt_step 
                        K1_vy  = ( -ωyLi**2 * Yt_[jt-1] + MFHam_vy(Xt_[jt-1],Yt_[jt-1],Zt_[jt-1], Vxt_[jt-1],Vyt_[jt-1],Vzt_[jt-1], DetmG1 ) ) * dt_step 
                        K1_vz  = ( -ωzLi**2 * Zt_[jt-1] + MFHam_vz(Xt_[jt-1],Yt_[jt-1],Zt_[jt-1], Vxt_[jt-1],Vyt_[jt-1],Vzt_[jt-1], DetmG1 ) ) * dt_step 
                        #------# K2 coefficients  (including dt_step!)
                        DetmG2 = DetmG(δB_AXIS[jB],Xt_[jt-1]+K1_x/2, Zt_[jt-1]+K1_z/2)  # calculate once to save time
                        K2_x   = ( (Vxt_[jt-1]+K1_vx/2) + MFHam_x((Xt_[jt-1]+K1_x/2),(Yt_[jt-1]+K1_y/2),(Zt_[jt-1]+K1_z/2), \
                                                                  (Vxt_[jt-1]+K1_vx/2),(Vyt_[jt-1]+K1_vy/2),(Vzt_[jt-1]+K1_vz/2), DetmG2 ) ) * dt_step
                        K2_y   = ( (Vyt_[jt-1]+K1_vy/2) + MFHam_y((Xt_[jt-1]+K1_x/2),(Yt_[jt-1]+K1_y/2),(Zt_[jt-1]+K1_z/2), \
                                                                  (Vxt_[jt-1]+K1_vx/2),(Vyt_[jt-1]+K1_vy/2),(Vzt_[jt-1]+K1_vz/2), DetmG2 ) ) * dt_step
                        K2_z   = ( (Vzt_[jt-1]+K1_vz/2) + MFHam_z((Xt_[jt-1]+K1_x/2),(Yt_[jt-1]+K1_y/2),(Zt_[jt-1]+K1_z/2), \
                                                                  (Vxt_[jt-1]+K1_vx/2),(Vyt_[jt-1]+K1_vy/2),(Vzt_[jt-1]+K1_vz/2), DetmG2 ) ) * dt_step
                        K2_vx  = ( -ωxLi**2 * (Xt_[jt-1]+K1_x/2) + MFHam_vx((Xt_[jt-1]+K1_x/2),(Yt_[jt-1]+K1_y/2),(Zt_[jt-1]+K1_z/2), \
                                                                            (Vxt_[jt-1]+K1_vx/2),(Vyt_[jt-1]+K1_vy/2),(Vzt_[jt-1]+K1_vz/2), DetmG2 ) ) * dt_step
                        K2_vy  = ( -ωyLi**2 * (Yt_[jt-1]+K1_y/2) + MFHam_vy((Xt_[jt-1]+K1_x/2),(Yt_[jt-1]+K1_y/2),(Zt_[jt-1]+K1_z/2), \
                                                                            (Vxt_[jt-1]+K1_vx/2),(Vyt_[jt-1]+K1_vy/2),(Vzt_[jt-1]+K1_vz/2), DetmG2 ) ) * dt_step
                        K2_vz  = ( -ωzLi**2 * (Zt_[jt-1]+K1_z/2) + MFHam_vz((Xt_[jt-1]+K1_x/2),(Yt_[jt-1]+K1_y/2),(Zt_[jt-1]+K1_z/2), \
                                                                            (Vxt_[jt-1]+K1_vx/2),(Vyt_[jt-1]+K1_vy/2),(Vzt_[jt-1]+K1_vz/2), DetmG2 ) ) * dt_step
                        #------# K3 coefficients  (including dt_step!)
                        DetmG3 = DetmG(δB_AXIS[jB],Xt_[jt-1]+K2_x/2, Zt_[jt-1]+K2_z/2)  # calculate once to save time
                        K3_x   = ( (Vxt_[jt-1]+K2_vx/2) + MFHam_x((Xt_[jt-1]+K2_x/2),(Yt_[jt-1]+K2_y/2),(Zt_[jt-1]+K2_z/2), \
                                                                  (Vxt_[jt-1]+K2_vx/2),(Vyt_[jt-1]+K2_vy/2),(Vzt_[jt-1]+K2_vz/2), DetmG3 ) ) * dt_step
                        K3_y   = ( (Vyt_[jt-1]+K2_vy/2) + MFHam_y((Xt_[jt-1]+K2_x/2),(Yt_[jt-1]+K2_y/2),(Zt_[jt-1]+K2_z/2), \
                                                                  (Vxt_[jt-1]+K2_vx/2),(Vyt_[jt-1]+K2_vy/2),(Vzt_[jt-1]+K2_vz/2), DetmG3 ) ) * dt_step
                        K3_z   = ( (Vzt_[jt-1]+K2_vz/2) + MFHam_z((Xt_[jt-1]+K2_x/2),(Yt_[jt-1]+K2_y/2),(Zt_[jt-1]+K2_z/2), \
                                                                  (Vxt_[jt-1]+K2_vx/2),(Vyt_[jt-1]+K2_vy/2),(Vzt_[jt-1]+K2_vz/2), DetmG3 ) ) * dt_step
                        K3_vx  = ( -ωxLi**2 * (Xt_[jt-1]+K2_x/2) + MFHam_vx((Xt_[jt-1]+K2_x/2),(Yt_[jt-1]+K2_y/2),(Zt_[jt-1]+K2_z/2), \
                                                                            (Vxt_[jt-1]+K2_vx/2),(Vyt_[jt-1]+K2_vy/2),(Vzt_[jt-1]+K2_vz/2), DetmG3 ) ) * dt_step
                        K3_vy  = ( -ωyLi**2 * (Yt_[jt-1]+K2_y/2) + MFHam_vy((Xt_[jt-1]+K2_x/2),(Yt_[jt-1]+K2_y/2),(Zt_[jt-1]+K2_z/2), \
                                                                            (Vxt_[jt-1]+K2_vx/2),(Vyt_[jt-1]+K2_vy/2),(Vzt_[jt-1]+K2_vz/2), DetmG3 ) ) * dt_step
                        K3_vz  = ( -ωzLi**2 * (Zt_[jt-1]+K2_z/2) + MFHam_vz((Xt_[jt-1]+K2_x/2),(Yt_[jt-1]+K2_y/2),(Zt_[jt-1]+K2_z/2), \
                                                                            (Vxt_[jt-1]+K2_vx/2),(Vyt_[jt-1]+K2_vy/2),(Vzt_[jt-1]+K2_vz/2), DetmG3 ) ) * dt_step
                        #------# K4 coefficients  (including dt_step!)
                        DetmG4 = DetmG(δB_AXIS[jB],Xt_[jt-1]+K3_x, Zt_[jt-1]+K3_z)  # calculate once to save time
                        K4_x   = ( (Vxt_[jt-1]+K3_vx) + MFHam_x((Xt_[jt-1]+K3_x),(Yt_[jt-1]+K3_y),(Zt_[jt-1]+K3_z), \
                                                                (Vxt_[jt-1]+K3_vx),(Vyt_[jt-1]+K3_vy),(Vzt_[jt-1]+K3_vz), DetmG4 ) ) * dt_step
                        K4_y   = ( (Vyt_[jt-1]+K3_vy) + MFHam_y((Xt_[jt-1]+K3_x),(Yt_[jt-1]+K3_y),(Zt_[jt-1]+K3_z), \
                                                                (Vxt_[jt-1]+K3_vx),(Vyt_[jt-1]+K3_vy),(Vzt_[jt-1]+K3_vz), DetmG4 ) ) * dt_step
                        K4_z   = ( (Vzt_[jt-1]+K3_vz) + MFHam_z((Xt_[jt-1]+K3_x),(Yt_[jt-1]+K3_y),(Zt_[jt-1]+K3_z), \
                                                                (Vxt_[jt-1]+K3_vx),(Vyt_[jt-1]+K3_vy),(Vzt_[jt-1]+K3_vz), DetmG4 ) ) * dt_step
                        K4_vx  = ( -ωxLi**2 * (Xt_[jt-1]+K3_x) + MFHam_vx((Xt_[jt-1]+K3_x),(Yt_[jt-1]+K3_y),(Zt_[jt-1]+K3_z), \
                                                                          (Vxt_[jt-1]+K3_vx),(Vyt_[jt-1]+K3_vy),(Vzt_[jt-1]+K3_vz), DetmG4 ) ) * dt_step
                        K4_vy  = ( -ωyLi**2 * (Yt_[jt-1]+K3_y) + MFHam_vy((Xt_[jt-1]+K3_x),(Yt_[jt-1]+K3_y),(Zt_[jt-1]+K3_z), \
                                                                          (Vxt_[jt-1]+K3_vx),(Vyt_[jt-1]+K3_vy),(Vzt_[jt-1]+K3_vz), DetmG4 ) ) * dt_step
                        K4_vz  = ( -ωzLi**2 * (Zt_[jt-1]+K3_z) + MFHam_vz((Xt_[jt-1]+K3_x),(Yt_[jt-1]+K3_y),(Zt_[jt-1]+K3_z), \
                                                                          (Vxt_[jt-1]+K3_vx),(Vyt_[jt-1]+K3_vy),(Vzt_[jt-1]+K3_vz), DetmG4 ) ) * dt_step
                        #------# 
                        #------# Propagation with Runge-Kutta
                        Xt_[jt]  = Xt_[jt-1] + (K1_x + 2*K2_x + 2*K3_x + K4_x)/6.
                        Yt_[jt]  = Yt_[jt-1] + (K1_y + 2*K2_y + 2*K3_y + K4_y)/6.
                        Zt_[jt]  = Zt_[jt-1] + (K1_z + 2*K2_z + 2*K3_z + K4_z)/6.
                        Vxt_[jt] = Vxt_[jt-1] + (K1_vx + 2*K2_vx + 2*K3_vx + K4_vx)/6.
                        Vyt_[jt] = Vyt_[jt-1] + (K1_vy + 2*K2_vy + 2*K3_vy + K4_vy)/6.
                        Vzt_[jt] = Vzt_[jt-1] + (K1_vz + 2*K2_vz + 2*K3_vz + K4_vz)/6.
                        #------#
                    
                    #----------# Mean field à la Dima:  (to be adjusted)
                    if (MeanFieldDima == 1):
                        #----# Generate gaussrand Cr velocity  (independent from collision one)      
                        VxCr   = random.gauss(0.0, SV0x_Cr)
                        VyCr   = random.gauss(0.0, SV0_Cr)
                        VzCr   = random.gauss(0.0, SV0_Cr)
                        #----# Relative velocity
                        Vrelx  = Vxt_[jt] - VxCr  
                        Vrely  = Vyt_[jt] - VyCr  
                        Vrelz  = Vzt_[jt] - VzCr   
                        Vrelt  = sumq3(Vrelx, Vrely, Vrelz)
                        #----# Velocity change 
                        Vxt_[jt] = Vxt_[jt] - dEMF_dx(Xt_[jt],Yt_[jt],Zt_[jt], (m_red*Vrelt/h_bar), DetmG(δB_AXIS[jB],Xt_[jt],Zt_[jt]) )/m_Li6 * dt_step
                        Vyt_[jt] = Vyt_[jt] - dEMF_dy(Xt_[jt],Yt_[jt],Zt_[jt], (m_red*Vrelt/h_bar), DetmG(δB_AXIS[jB],Xt_[jt],Zt_[jt]) )/m_Li6 * dt_step
                        Vzt_[jt] = Vzt_[jt] - dEMF_dz(Xt_[jt],Yt_[jt],Zt_[jt], (m_red*Vrelt/h_bar), DetmG(δB_AXIS[jB],Xt_[jt],Zt_[jt]) )/m_Li6 * dt_step
                        #----# Re-compute motion parameters 
                        Analφx[ja] = ( m.atan2((ωxLi*Xt_[jt]), Vxt_[jt]) - ωxLi*timejt_ ) % twoπ
                        Analφy[ja] = ( m.atan2((ωyLi*Yt_[jt]), Vyt_[jt]) - ωyLi*timejt_ ) % twoπ
                        Analφz[ja] = ( m.atan2((ωzLi*Zt_[jt]), Vzt_[jt]) - ωzLi*timejt_ ) % twoπ
                        AnalAx[ja] = Vxt_[jt]/(ωxLi*m.cos(ωxLi*timejt_ + Analφx[ja]))
                        AnalAy[ja] = Vyt_[jt]/(ωyLi*m.cos(ωyLi*timejt_ + Analφy[ja]))
                        AnalAz[ja] = Vzt_[jt]/(ωzLi*m.cos(ωzLi*timejt_ + Analφz[ja]))   
                    
                    #-------------------------------------------# {{{ Collisions }}}
                    randcoll = random.uniform(0.,1.)   
                    if   (modecoll == 'Cr_M∞'):
                        VLitot_  = m.sqrt(Vxt_[jt]*Vxt_[jt] + Vyt_[jt]*Vyt_[jt] + Vzt_[jt]*Vzt_[jt])  
                        Gamma_dt = Γel_vLi(DetmG(δB_AXIS[jB],Xt_[jt],Zt_[jt]), VLitot_, n_Gauss(crbath*Xt_[jt],crbath*Yt_[jt],crbath*Zt_[jt], Natom_Cr,sx_Cr,sy_Cr,sz_Cr)) * dt_step  
                    elif (modecoll == 'Cr_RealT'):
                        #----# Generate gaussrand Cr velocity           
                        VxCr   = random.gauss(0.0, SV0x_Cr)
                        VyCr   = random.gauss(0.0, SV0_Cr)
                        VzCr   = random.gauss(0.0, SV0_Cr)
                        if (SameCrcoll == 1 and RecentColl != 0):
                            VxCr = VXCrClose
                            VyCr = VYCrClose
                            VzCr = VZCrClose
                        
                        #----# Relative velocity  (before collision)  # this is needed even if (VRELcoll == 'SigmaVCr')
                        Vrelx  = Vxt_[jt] - VxCr  
                        Vrely  = Vyt_[jt] - VyCr  
                        Vrelz  = Vzt_[jt] - VzCr 
                        Vrelt  = m.sqrt(Vrelx*Vrelx + Vrely*Vrely + Vrelz*Vrelz)
                        Vrelt_Γdt = Vrelt
                        if (VRELcoll == 'SigmaVCr'):  
                            Vrelx_Γdt  = m.sqrt(Vxt_[jt]*Vxt_[jt] + SV0x_Cr*SV0x_Cr)  
                            Vrely_Γdt  = m.sqrt(Vyt_[jt]*Vyt_[jt] + SV0_Cr*SV0_Cr)
                            Vrelz_Γdt  = m.sqrt(Vzt_[jt]*Vzt_[jt] + SV0_Cr*SV0_Cr)  
                            Vrelt_Γdt  = m.sqrt(Vrelx_Γdt*Vrelx_Γdt + Vrely_Γdt*Vrely_Γdt + Vrelz_Γdt*Vrelz_Γdt)                     
                        
                        #----#
                        Gamma_dt = Γel_vrel(DetmG(δB_AXIS[jB],Xt_[jt],Zt_[jt]), Vrelt_Γdt, n_Gauss(crbath*Xt_[jt],crbath*Yt_[jt],crbath*Zt_[jt], Natom_Cr,sx_Cr,sy_Cr,sz_Cr)) * dt_step
                    
                    #-----#
                    Gammadt_avg[jB] = Gammadt_avg[jB] + Gamma_dt
                    if (Gamma_dt > 1.0):
                        FLAG_Gmdt[jB] = FLAG_Gmdt[jB] + 1     # check that Gamma*dt is less than 1, otherwise FLAG
                    
                    if (Gamma_dt > Gammadt_max[jB]):
                        Gammadt_max[jB] = Gamma_dt 
                    
                    #-----# Prevent collisions during "trapping" retardation time
                    if (BlockRetColls == 1 and jt_prevcoll != 0):
                        timefromlastcoll = timejt_ - timejt(jt_prevcoll)
                        if (timefromlastcoll < Retard_Time):
                            EnableColl = 0
                        elif (timefromlastcoll < Retard_Time + dt_step):
                            EnableColl = (timefromlastcoll - Retard_Time)/dt_step
                        else:
                            EnableColl = 1
                        
                    
                    #-----#
                    #----------------------# COLLISION STUFF
                    if (randcoll < EnableColl*Gamma_dt):   #  --> ((Collision happening!))
                        #---# Update counters, calculate MFP stuff
                        Ncoll_atom[ja] = Ncoll_atom[ja] + 1
                        Δt_free        = (1.e3) * (jt - jt_prevcoll) * dt_step        # [ms]
                        Δx_free        = (1.e6) * m.fabs(Xt_[jt] - Xt_[jt_prevcoll])  # [µm]
                        RealΔx_free    = (1.e6) * sum([(m.fabs(Xt_[qx+1] - Xt_[qx])) for qx in range(jt_prevcoll, jt)])   # [µm]   
                        Real_λ_free    = (1.e6) * sum([(sumq3((Xt_[qx+1]-Xt_[qx]), (Yt_[qx+1]-Yt_[qx]), (Zt_[qx+1]-Zt_[qx]))) for qx in range(jt_prevcoll, jt)])  # [µm]     
                        jt_prevcoll    = jt
                        krel_RetTime   = (mred_hbar*Vrelt) if (modecoll == 'Cr_RealT') else (mLi6_hbar*VLitot_)
                        Retard_Time    = RetardTime(DetmG(δB_AXIS[jB],Xt_[jt],Zt_[jt]), krel_RetTime)     # [s], but then saved in [µs]
                        RecentColl     = (1) if (RecentColl == 0) else (2)    # for correlated collisions
                        #---# Print in mfp
                        print("%.4f %.3f %.3f %.3f %.3f" % (Δt_free, Δx_free, RealΔx_free, Real_λ_free, (1.e6)*Retard_Time), file=mfp)
                        #---# s-wave collision (now good!)
                        #randTheta     = RandNum(0., π)            # ((NB)):  WRONG!!
                        randcosTheta   = random.uniform(-1.,1.)    # = z projection, flat
                        randTheta      = m.acos(randcosTheta)
                        randPhi        = random.uniform(0., 2*π)   # This one instead is fine, indeed is actually good.    
                        #---#
                        if   (modecoll == 'Cr_M∞'):       # Cr has infinite mass, it does not recoil nor transfer/absorb energy from Li, AND IT'S NOT MOVING AT ALL
                            Vxt_[jt]  = VLitot_ * m.sin(randTheta)*m.cos(randPhi)
                            Vyt_[jt]  = VLitot_ * m.sin(randTheta)*m.sin(randPhi)
                            Vzt_[jt]  = VLitot_ * m.cos(randTheta)             
                        elif (modecoll == 'Cr_RealT'):    # Cr has FINITE mass and a Temperature, it DOES recoil and CAN transfer/absorb energy from Li 
                            #----# After collision
                            Vrelx_ = Vrelt * m.sin(randTheta)*m.cos(randPhi)
                            Vrely_ = Vrelt * m.sin(randTheta)*m.sin(randPhi)
                            Vrelz_ = Vrelt * m.cos(randTheta)
                            #----# Li exit velocity  (OVERWRITE  Vxt_[jt])
                            Vxt_[jt] = (Vxt_[jt] + MassRat*(VxCr+Vrelx_)) / (1.+MassRat)    # checked, requiring momentum conservation
                            Vyt_[jt] = (Vyt_[jt] + MassRat*(VyCr+Vrely_)) / (1.+MassRat)
                            Vzt_[jt] = (Vzt_[jt] + MassRat*(VzCr+Vrelz_)) / (1.+MassRat)
                            #----# Cr exit velocity  (for repeated collisions)
                            if (SameCrcoll == 1):
                                VXCr_out = Vxt_[jt] - Vrelx_   
                                VYCr_out = Vyt_[jt] - Vrely_ 
                                VZCr_out = Vzt_[jt] - Vrelz_ 
                            
                            #----#
                        
                        #-------# Re-compute motion parameters (if no RetTime was there)  (this is needed anyway)
                        if (modeEvol == "analytic"):
                            Analφx[ja] = ( m.atan2((ωxLi*Xt_[jt]), Vxt_[jt]) - ωxLi*timejt_ ) % twoπ
                            Analφy[ja] = ( m.atan2((ωyLi*Yt_[jt]), Vyt_[jt]) - ωyLi*timejt_ ) % twoπ
                            Analφz[ja] = ( m.atan2((ωzLi*Zt_[jt]), Vzt_[jt]) - ωzLi*timejt_ ) % twoπ
                            AnalAx[ja] = Vxt_[jt]/(ωxLi*m.cos(ωxLi*timejt_ + Analφx[ja]))
                            AnalAy[ja] = Vyt_[jt]/(ωyLi*m.cos(ωyLi*timejt_ + Analφy[ja]))
                            AnalAz[ja] = Vzt_[jt]/(ωzLi*m.cos(ωzLi*timejt_ + Analφz[ja]))
                            if (SameCrcoll == 1):
                                #-----# Cr position for the collision
                                XCrColl   = (Xt_[jt]) if (RecentColl == 1) else (XCrClose)
                                YCrColl   = (Yt_[jt]) if (RecentColl == 1) else (YCrClose)
                                ZCrColl   = (Zt_[jt]) if (RecentColl == 1) else (ZCrClose)
                                #-----# Motion params for Cr atom that has collided
                                Analφx_Cr = ( m.atan2((ωxCr*XCrColl), VXCr_out) - ωxCr*timejt_ ) % twoπ
                                Analφy_Cr = ( m.atan2((ωyCr*YCrColl), VYCr_out) - ωyCr*timejt_ ) % twoπ
                                Analφz_Cr = ( m.atan2((ωzCr*ZCrColl), VZCr_out) - ωzCr*timejt_ ) % twoπ
                                AnalAx_Cr = VXCr_out/(ωxCr*m.cos(ωxCr*timejt_ + Analφx_Cr))
                                AnalAy_Cr = VYCr_out/(ωyCr*m.cos(ωyCr*timejt_ + Analφy_Cr))
                                AnalAz_Cr = VZCr_out/(ωzCr*m.cos(ωzCr*timejt_ + Analφz_Cr))   
                            
                        elif (modeEvol == "numericMF" and RetTimePhase == 2):
                            Liφx_ = ( m.atan2((ωxLi*Xt_[jt]), Vxt_[jt]) - ωxLi*timejt_ ) % twoπ
                            Liφy_ = ( m.atan2((ωyLi*Yt_[jt]), Vyt_[jt]) - ωyLi*timejt_ ) % twoπ
                            Liφz_ = ( m.atan2((ωzLi*Zt_[jt]), Vzt_[jt]) - ωzLi*timejt_ ) % twoπ
                            LiAx_ = Vxt_[jt]/(ωxLi*m.cos(ωxLi*timejt_ + Liφx_))
                            LiAy_ = Vyt_[jt]/(ωyLi*m.cos(ωyLi*timejt_ + Liφy_))
                            LiAz_ = Vzt_[jt]/(ωzLi*m.cos(ωzLi*timejt_ + Liφz_))
                        
                        #-------# Retard time correction on phase (after having computed amplitude)  --> SHIFT ON LI
                        if (RetTimePhase == 1 and modeEvol == "analytic"):
                            Analφx[ja] = ( Analφx[ja] - ωxLi*Retard_Time ) % twoπ
                            Analφy[ja] = ( Analφy[ja] - ωyLi*Retard_Time ) % twoπ
                            Analφz[ja] = ( Analφz[ja] - ωzLi*Retard_Time ) % twoπ
                        
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
                            timejt_RT = timejt_ + Retard_Time
                            if   (modeEvol == "analytic"):
                                xLi__  = AnalAx[ja] * m.sin(ωxLi*timejt_RT + Analφx[ja])
                                yLi__  = AnalAy[ja] * m.sin(ωyLi*timejt_RT + Analφy[ja])
                                zLi__  = AnalAz[ja] * m.sin(ωzLi*timejt_RT + Analφz[ja])
                                vxLi__ = (ωxLi*AnalAx[ja]) * m.cos(ωxLi*timejt_RT + Analφx[ja])
                                vyLi__ = (ωyLi*AnalAy[ja]) * m.cos(ωyLi*timejt_RT + Analφy[ja])
                                vzLi__ = (ωzLi*AnalAz[ja]) * m.cos(ωzLi*timejt_RT + Analφz[ja])
                            elif (modeEvol == "numericMF"):
                                xLi__  = LiAx_ * m.sin(ωxLi*timejt_RT + Liφx_)
                                yLi__  = LiAy_ * m.sin(ωyLi*timejt_RT + Liφy_)
                                zLi__  = LiAz_ * m.sin(ωzLi*timejt_RT + Liφz_)
                                vxLi__ = ωxLi*LiAx_ * m.cos(ωxLi*timejt_RT + Liφx_)
                                vyLi__ = ωyLi*LiAy_ * m.cos(ωyLi*timejt_RT + Liφy_)
                                vzLi__ = ωzLi*LiAz_ * m.cos(ωzLi*timejt_RT + Liφz_)
                            
                            #----# Cr motion params (after collision, if no retard time)     [for c.o.m.]
                            Crφx_  = ( m.atan2((ωxCr*Xt_[jt]), VxCr_) - ωxCr*timejt_ ) % twoπ
                            Crφy_  = ( m.atan2((ωyCr*Yt_[jt]), VyCr_) - ωyCr*timejt_ ) % twoπ
                            Crφz_  = ( m.atan2((ωzCr*Zt_[jt]), VzCr_) - ωzCr*timejt_ ) % twoπ
                            CrAx_  = VxCr_/(ωxCr*m.cos(ωxCr*timejt_ + Crφx_))
                            CrAy_  = VyCr_/(ωyCr*m.cos(ωyCr*timejt_ + Crφy_))
                            CrAz_  = VzCr_/(ωzCr*m.cos(ωzCr*timejt_ + Crφz_))
                            #----# Cr "unperturbed" position/velocity at time (t* + Δt_ret)  [for c.o.m.]
                            xCr__  = CrAx_ * m.sin(ωxCr*timejt_RT + Crφx_)
                            yCr__  = CrAy_ * m.sin(ωyCr*timejt_RT + Crφy_)
                            zCr__  = CrAz_ * m.sin(ωzCr*timejt_RT + Crφz_)
                            vxCr__ = ωxCr*CrAx_ * m.cos(ωxCr*timejt_RT + Crφx_)
                            vyCr__ = ωyCr*CrAy_ * m.cos(ωyCr*timejt_RT + Crφy_)
                            vzCr__ = ωzCr*CrAz_ * m.cos(ωzCr*timejt_RT + Crφz_)
                            #----# C.o.M. "unperturbed" position/velocity at time (t* + Δt_ret)
                            xcom__ = (m_Li6*xLi__ + m_Cr53*xCr__)/m_LiCr   
                            ycom__ = (m_Li6*yLi__ + m_Cr53*yCr__)/m_LiCr
                            zcom__ = (m_Li6*zLi__ + m_Cr53*zCr__)/m_LiCr
                            vxcom__= (m_Li6*vxLi__ + m_Cr53*vxCr__)/m_LiCr            
                            vycom__= (m_Li6*vyLi__ + m_Cr53*vyCr__)/m_LiCr 
                            vzcom__= (m_Li6*vzLi__ + m_Cr53*vzCr__)/m_LiCr 
                            #----# Fix passage at time (t* + Δt_ret) in new Li trajectory
                            xListar  = xcom__  # relative distance = 0
                            yListar  = ycom__
                            zListar  = zcom__
                            VxListar = vxcom__ + MassRat/(1.+MassRat) * Vrelx_
                            VyListar = vycom__ + MassRat/(1.+MassRat) * Vrely_
                            VzListar = vzcom__ + MassRat/(1.+MassRat) * Vrelz_
                            #----# New Motion params (final ones)
                            if   (modeEvol == "analytic"):
                                Analφx[ja] = ( m.atan2((ωxLi*xListar), VxListar) - ωxLi*timejt_RT ) % twoπ
                                Analφy[ja] = ( m.atan2((ωyLi*yListar), VyListar) - ωyLi*timejt_RT ) % twoπ
                                Analφz[ja] = ( m.atan2((ωzLi*zListar), VzListar) - ωzLi*timejt_RT ) % twoπ
                                AnalAx[ja] = VxListar/(ωxLi*m.cos(ωxLi*timejt_RT + Analφx[ja]))
                                AnalAy[ja] = VyListar/(ωyLi*m.cos(ωyLi*timejt_RT + Analφy[ja]))
                                AnalAz[ja] = VzListar/(ωzLi*m.cos(ωzLi*timejt_RT + Analφz[ja]))
                            elif (modeEvol == "numericMF"):
                                Newφx = ( m.atan2((ωxLi*xListar), VxListar) - ωxLi*timejt_RT ) % twoπ
                                Newφy = ( m.atan2((ωyLi*yListar), VyListar) - ωyLi*timejt_RT ) % twoπ
                                Newφz = ( m.atan2((ωzLi*zListar), VzListar) - ωzLi*timejt_RT ) % twoπ
                                NewAx = VxListar/(ωxLi*m.cos(ωxLi*timejt_RT + Newφx))
                                NewAy = VyListar/(ωyLi*m.cos(ωyLi*timejt_RT + Newφy))
                                NewAz = VzListar/(ωzLi*m.cos(ωzLi*timejt_RT + Newφz))
                                #-----#
                                Xt_[jt]  = NewAx * m.sin(ωxLi*timejt_ + Newφx)    # NB: timejt_ (w/o RT) here! RT is already in Newφ
                                Yt_[jt]  = NewAy * m.sin(ωyLi*timejt_ + Newφy)
                                Zt_[jt]  = NewAz * m.sin(ωzLi*timejt_ + Newφz)
                                Vxt_[jt] = ωxLi*NewAx * m.cos(ωxLi*timejt_ + Newφx)
                                Vyt_[jt] = ωyLi*NewAy * m.cos(ωyLi*timejt_ + Newφy)
                                Vzt_[jt] = ωzLi*NewAz * m.cos(ωzLi*timejt_ + Newφz)
                            
                            #----#
                        
                        #-------#
                    
                    #----------------------#
                    #-------------------------------------------# {{{ END Collisions }}}  
                    #-----#
                    if (BFieldDrift != 0):
                        δB_AXIS[jB] = Central_δB    # Restore 
                    
                    #-----#
                
                #-------------------------------------# [[[ END Trajectory cycle]]]
                #-----------------------#
                #-----------------------#
                #-----# Calculate (EnergyCons)   ---> print moved outside ATOM cycle, before END DET cycle. E_tot arrays are NOT overwritten
                Epot_fin[ja] = 0.5*m_Li6*sumq3((ωxLi*Xt_[N_tsteps-1]), (ωyLi*Yt_[N_tsteps-1]), (ωzLi*Zt_[N_tsteps-1]))**2
                Ekin_fin[ja] = 0.5*m_Li6*(Vxt_[N_tsteps-1]*Vxt_[N_tsteps-1] + Vyt_[N_tsteps-1]*Vyt_[N_tsteps-1] + Vzt_[N_tsteps-1]*Vzt_[N_tsteps-1])  
                Etot_fin[ja] = Epot_fin[ja] + Ekin_fin[ja]           
                #-----# Print (Coords05)  --> NB:  Xt_,... arrays are OVERWRITTEN!  but opening can be moved
                #with open(Coords05_det(SimSet,jB), 'a') as coords05:
                for j05 in range(N_05+1):
                    jmult05_ = jCorrespTo_time(t05_from_j05(j05))
                    print("%.2f %d %.1f %.3f %.3f %.3f %.3f %.3f %.3f" \
                        % (δB_AXIS[jB], ja+1, t05_from_j05(j05), \
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
        
    
    #--------# Print EnergyCons  (NEW position!)
    with open(EnergyCons_det(SimSet,jB), 'a') as energycons:
        for ja in range(N_atoms):  
            print(δB_AXIS[jB], ja+1, Etot_in[ja], Etot_fin[ja], Ncoll_atom[ja], \
                  Epot_in[ja], Ekin_in[ja], Epot_fin[ja], Ekin_fin[ja], file=energycons)
        
    
    #--------#
    #--------#
    Gammadt_avg[jB]    = Gammadt_avg[jB]/(N_tsteps-1.)/N_atoms
    PreGammadt_avg[jB] = PreGammadt_avg[jB]/(N_presteps-1.)/N_atoms
    #--------# Save avg number of collisions at this detuning
    Ncoll_avg[jB]    = sum([(Ncoll_atom[nn]) for nn in range(N_atoms)]) / N_atoms
    PreNcoll_avg[jB] = sum([(PreNcoll_atom[nn]) for nn in range(N_atoms)]) / N_atoms
    #Ncoll_avg_e[jB]  = STATS_stddev  # maybe STATS_mean_err is better in this case?
    #Ncoll_avg_e2[jB] = STATS_mean_err
    #--------#


#--------------------------------------------------#  [[[END Detuning cycle]]]
#------------------------------------#
#----------------------#
#--------#

#--------# Save avg Ncoll [for this instance]
with open(AvgNcoll_inst(SimSet,SimInstn), 'w') as avgncoll:
    print("Det(mG)\t Ncoll_avg \t PreNcoll_avg", file=avgncoll)
    for jB in range(N_ptsδB):
        print(δB_AXIS[jB], "\t", Ncoll_avg[jB], "\t", PreNcoll_avg[jB], file=avgncoll)

#--------# Save FLAGScoll [for this instance]
with open(FLAGScoll_inst(SimSet,SimInstn), 'w') as flagscoll:
    print("Det(mG)\tNFLAGS_tot\tGammadt_max\tGammadt_avg\tdt_step(µs)\tPreNFLAGS_tot\tPreGammadt_max\tPreGammadt_avg", file=flagscoll)
    for jB in range(N_ptsδB):
        print( δB_AXIS[jB], "\t", FLAG_Gmdt[jB], "\t", Gammadt_max[jB], "\t",  \
               Gammadt_avg[jB], "\t", 1.e6*(DTSTEP[IsCloseToRes_dt(δB_AXIS[jB])]), "\t", \
               PreFLAG_Gmdt[jB], "\t", PreGammadt_max[jB], "\t", PreGammadt_avg[jB], file=flagscoll)

#--------#

#--------# Save AvgDOTS [for this instance]
#with open(AvgDOTS_inst(SimSet,SimInstn), 'w') as avgdots:
#    print("Det(mG)\txdot_avg(µm)\tydot_avg(µm)\tzdot_avg(µm)\tvxdot_avg(µm/ms)\tvydot_avg(µm/ms)\tvzdot_avg(µm/ms)\tdt_step(µs)", file=avgdots)
#    for jB in range(N_ptsδB):
#        print( δB_AXIS[jB],   "\t", xdot_avg[jB],  "\t", ydot_avg[jB], "\t",  zdot_avg[jB], "\t", \
#               vxdot_avg[jB], "\t", vydot_avg[jB], "\t", vzdot_avg[jB], "\t", 1.e6*(DTSTEP[IsCloseToRes_dt(δB_AXIS[jB])]), file=avgdots)

#--------#



