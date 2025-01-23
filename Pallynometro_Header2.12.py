#-------------------------------------------#
#--------#     Pinball  Machine    #--------#
#--------#      ...on Python!      #--------#
#--------#       [[Header]]        #--------#                 
#-------------------------------------------#
import math as m
import numpy
import random
#-----#

#------------------------------#
#-----#  [FoldersFiles]  #-----#
#------------------------------#
phdpath         = "C:/Users/stefa/Desktop/PhD/"
phddata         = phdpath + "Data/"
SpinDiffCross   = phddata + "CrLi FermiMix/Spin Diffusion/"
SpinDiff_SimDat = SpinDiffCross + "SimulationsData/"
#-----#
def Subfold(simset):
    return SpinDiff_SimDat + "Set " + simset + "/"

def EnergyCons_det(simset,jB):
    return Subfold(simset) + "energycons/EnergyCons_%.1f_mG.dat" % (δB_AXIS[jB])

def Coords05_det(simset,jB):
    return Subfold(simset) + "coords05/Coordinates_tmult0.5_%.1f_mG.dat" % (δB_AXIS[jB])

def MeanFreePath_det(simset,jB):
    return Subfold(simset) + "meanfreepath/MFP_%.1f_mG.dat" % (δB_AXIS[jB])

def AvgNcoll_inst(simset,inst):
    return Subfold(simset) + "AvgNcoll_%d.dat" % (int(inst))

def FLAGScoll_inst(simset,inst):
    return Subfold(simset) + "FLAGScoll_%d.dat" % (int(inst))

def AvgDOTS_inst(simset,inst):
    return Subfold(simset) + "AvgDOTS_%d.dat" % (int(inst))

#-----#

#------------------------------#


#------------------------------#
#-----#  [Math MiniLib]  #-----#
#------------------------------#
#-----# Numbers
π = m.pi
ι = 1j
#-----# Rename basic math functions
def abs(x):
    return m.fabs(x)

def exp(x):
    return m.exp(x)

def sin(x):
    return m.sin(x)

def cos(x):
    return m.cos(x)

def acos(x):
    return m.acos(x)

def atan2(y,x):
    return m.atan2(y,x)

def sqrt(x):
    return m.sqrt(x)

#-----# Rename Random functions
def RandNum(a, b):
    return random.uniform(a, b)

def GaussRandNum(m, s):
    return random.gauss(m, s)

#-----# MyFunctions
def sumq2(a,b):
    return m.sqrt(a**2 + b**2)

def sumq3(a,b,c):
    return m.sqrt(a**2 + b**2 + c**2)

def mod2π(x):
    return x%(2*π)

def WeightAvg2(x1,w1, x2,w2):
    return (w1*x1 + w2*x2)/(w1 + w2)

#-----#

#------------------------------#



#------------------------------#
#-----#  [PhysicsConst]  #-----# 
#------------------------------#
#-----# Physics
ccc = 2.99792458        * 1.e8     # [m/s]    Speed of light
eee = 1.60217663        * 1.e-19   # [C]      Elementary charge
k_B = 1.380649          * 1.e-23   # [J/K]    Boltzmann
h_p = 6.62607004        * 1.e-34   # [J·s]    Planck
ħ   = h_bar = h_p/(2*π)            # [J·s]    Reduced Planck
m_p = 1.6726219         * 1.e-27   # [kg]     Proton mass
m_e = 9.1093837         * 1.e-31   # [kg]     Electron mass
ε0  = 8.8541878128      * 1.e-12   # [F/m]
#m_e = 9.10938356       * 1.e-31   # [kg]
gs  = 2.0023193043737              #          Electron g factor
a_Bohr = 5.29177210903  * 1.e-11   # [m]      Bohr radius (H)
µ_B    = 9.274009994    * 1.e-28   # [J/G]    Bohr magneton
#-----#

#-----# Li-Cr numbers
m_Li6  = 6.015122795 * m_p   # mass  [kg]   # OLD
m_Cr53 = 52.940647   * m_p   # mass  [kg]
m_red  = (m_Li6*m_Cr53)/(m_Li6+m_Cr53)
#-----# LiCr 2-body scattering params  [from Li2-Cr1 @ 1461 G]
abg_1461G =   41.48   # [a_0]
ΔB_1461G  =  476.56   #  [mG]
Rs0_1461G = 6017.66   # [a_0]
δµ_1461G  = 1.988     # [µ_B]
#-----#

#-----# Pre-calculation of consts
mred_hbar = m_red/h_bar
mLi6_hbar = m_Li6/h_bar
mred_mLi6 = m_red/m_Li6
twoπ      = 2*π
fourπ     = 4*π
hbar_sqr  = h_bar**2
mLi6mred  = m_Li6*m_red
sqrt2π_3  = (m.sqrt(twoπ))**3
#------------------------------#


#------------------------------#
#-----#  [Physics Lib.]  #-----#
#------------------------------#

#-----------------------# Density distributions (Gauss)
def n0_Gauss(N0,sx,sy,sz):
    return N0 / (sqrt2π_3 * sx*sy*sz)

def n_Gauss(x,y,z,N0,sx,sy,sz):
    x_sx = x/sx
    y_sy = y/sy
    z_sz = z/sz
    return n0_Gauss(N0,sx,sy,sz) * m.exp(-0.5*(x_sx*x_sx + y_sy*y_sy + z_sz*z_sz))      

#-----------------------# Detuning dependent on position (curvature, gradient)
def DetmG(δB0,x,z):
    return δB0 + blevgrad*z - 0.5*Bcurv_x * x*x

#-----------------------# Scattering theory    NB:  check a_s, Rs  how defined
#--------# Scattering length and R*
def a_s(δB,a_bg,ΔB):
    #return -a_bg * ΔB/δB
    return a_bg * (1.0 - ΔB/δB)

def Rs(δB,Rs0,ΔB):
    #return Rs0 
    return Rs0  * (ΔB/(ΔB-δB))**2  

def as_1461G(δB):
    return a_s(δB,abg_1461G,ΔB_1461G)

def Rs_1461G(δB):
    return Rs(δB,Rs0_1461G,ΔB_1461G)

def as1461G(δB):
    return a_Bohr*a_s(δB,abg_1461G,ΔB_1461G)

def Rs1461G(δB):
    return a_Bohr*Rs(δB,Rs0_1461G,ΔB_1461G)

#--------# Scattering cross section (elastic, narrow res)
def σ_el(k,a_s,Rs):
    return (fourπ * a_s**2)/((1+Rs*a_s*k**2)**2 + (k*a_s)**2) 

#--------# Scattering rate           # ((NB)): no factor of 2π !
def Γ_el(n,σ,v):
    return (n * σ * v)  

def Γel_vLi(δB,vLi,ncoll):
    return Γ_el(ncoll, σ_el((m_Li6*vLi/h_bar), a_Bohr*as_1461G(δB), a_Bohr*Rs_1461G(δB)), vLi) 

def Γel_vrel(δB,vrel,ncoll):
    return Γ_el(ncoll, σ_el((mred_hbar*vrel), a_Bohr*as_1461G(δB), a_Bohr*Rs_1461G(δB)), vrel)

#--------# ScatteringAmplitude
def f0(k,a_s,Rs):
    return -1.0/(1./a_s + Rs*k**2 + k*1j)    # s-wave scattering amplitude

def f0_LiCr(δB,k_rel):
    return f0(k_rel, a_Bohr*as_1461G(δB), a_Bohr*Rs_1461G(δB))

#--------# RetardationTime (Wigner)
def dηdk(k,a_s,Rs):
    k2Rsas = k*k*Rs*a_s
    return a_s * (k2Rsas-1.)/((1.+k2Rsas)**2 + (k*a_s)**2)

def dηdk_LiCr(δB,k_rel):
    return dηdk(k_rel, a_Bohr*as_1461G(δB), a_Bohr*Rs_1461G(δB))

def RetardTime(δB,k_rel):
    return 2.*dηdk_LiCr(δB,k_rel)/(k_rel/mred_hbar)    # ((NB)): no factor of 2

#--------# MeanField  (à la Dima)
def E_MF(x,y,z,k_rel,δB):
    return -(2*π*ħ**2)/m_red * numpy.real(f0_LiCr(δB,k_rel)) * n_Gauss(x,y,z, Natom_Cr,sx_Cr,sy_Cr,sz_Cr)

def dEMF_dx(x,y,z,k_rel,δB):
    return -E_MF(x,y,z,k_rel,δB)*(x/sx_Cr**2)

def dEMF_dy(x,y,z,k_rel,δB):
    return -E_MF(x,y,z,k_rel,δB)*(y/sy_Cr**2)

def dEMF_dz(x,y,z,k_rel,δB):
    return -E_MF(x,y,z,k_rel,δB)*(z/sz_Cr**2)

#--------#
#-----------------------# 


#------------------------------#




#------------------------------#
#-----#  [MeanField Ham]  #----#  (Random V*)
#------------------------------#
#-----# relative k with "punctual" vLi and (random) VCr*
def krelMF(vx,vy,vz):
    VX_ = vx-VxCrMF
    VY_ = vy-VyCrMF
    VZ_ = vz-VzCrMF
    return mred_hbar * m.sqrt(VX_*VX_ + VY_*VY_ + VZ_*VZ_)

#-----# MF \dot{x}, with "punctual" vLi and (fixed) SigmaVCr
def MFHam_x(x,y,z,vx,vy,vz,δB):
    as_   = as1461G(δB)
    Rs_   = Rs1461G(δB)
    krel_ = krelMF(vx,vy,vz)
    return -fourπ * as_*as_*as_ * mred_mLi6 * n_Gauss(x,y,z,Natom_Cr,sx_Cr,sy_Cr,sz_Cr) * \
            (vx - VxCrMF) * (1.+ Rs_/as_ * (1.+ krel_*krel_ * Rs_*as_)**2)  \
            / ( (1.+ krel_*krel_ * Rs_*as_)**2 + (krel_*as_)**2 )**2

def MFHam_y(x,y,z,vx,vy,vz,δB):
    as_   = as1461G(δB)
    Rs_   = Rs1461G(δB)
    krel_ = krelMF(vx,vy,vz)
    return -fourπ * as_*as_*as_ * mred_mLi6 * n_Gauss(x,y,z,Natom_Cr,sx_Cr,sy_Cr,sz_Cr) * \
            (vy - VyCrMF) * (1.+ Rs_/as_ * (1.+ krel_*krel_ * Rs_*as_)**2)  \
            / ( (1.+ krel_*krel_ * Rs_*as_)**2 + (krel_*as_)**2 )**2

def MFHam_z(x,y,z,vx,vy,vz,δB):
    as_   = as1461G(δB)
    Rs_   = Rs1461G(δB)
    krel_ = krelMF(vx,vy,vz)
    return -fourπ * as_*as_*as_ * mred_mLi6 * n_Gauss(x,y,z,Natom_Cr,sx_Cr,sy_Cr,sz_Cr) * \
            (vz - VzCrMF) * (1.+ Rs_/as_ * (1.+ krel_*krel_ * Rs_*as_)**2)  \
            / ( (1.+ krel_*krel_ * Rs_*as_)**2 + (krel_*as_)**2 )**2

#-----# MF \dot{v}_x, with "punctual" vLi and (fixed) SigmaVCr.   NB:  divided by m_Li6 with respect to (H10)  [need eq for velocity]
def MFHam_vx(x,y,z,vx,vy,vz,δB):
    as_   = as1461G(δB)
    Rs_   = Rs1461G(δB)
    krel_ = krelMF(vx,vy,vz)
    return twoπ*as_*hbar_sqr / (mLi6mred*sx_Cr*sx_Cr) * n_Gauss(x,y,z,Natom_Cr,sx_Cr,sy_Cr,sz_Cr) * x * \
           (1.+ krel_*krel_ * Rs_*as_)/((1.+ krel_*krel_ * Rs_*as_)**2 + (krel_*as_)**2)

def MFHam_vy(x,y,z,vx,vy,vz,δB):
    as_   = as1461G(δB)
    Rs_   = Rs1461G(δB)
    krel_ = krelMF(vx,vy,vz)
    return twoπ*as_*hbar_sqr / (mLi6mred*sy_Cr*sy_Cr) * n_Gauss(x,y,z,Natom_Cr,sx_Cr,sy_Cr,sz_Cr) * y * \
          (1.+ krel_*krel_ * Rs_*as_)/((1.+ krel_*krel_ * Rs_*as_)**2 + (krel_*as_)**2)

def MFHam_vz(x,y,z,vx,vy,vz,δB):
    as_   = as1461G(δB)
    Rs_   = Rs1461G(δB)
    krel_ = krelMF(vx,vy,vz)
    return twoπ*as_*hbar_sqr / (mLi6mred*sz_Cr*sz_Cr) * n_Gauss(x,y,z,Natom_Cr,sx_Cr,sy_Cr,sz_Cr) * z * \
           (1.+ krel_*krel_ * Rs_*as_)/((1.+ krel_*krel_ * Rs_*as_)**2 + (krel_*as_)**2)

#------------------------------#




if (1):
    #------------------------------#
    #-----#  [MeanField Ham]  #----#  (ThermalAvg SigmaCr)
    #------------------------------#
    #-----# relative k with "punctual" vLi and thermal οv_Cr
    def krelMF(vx,vy,vz):
        return mred_hbar * m.sqrt(vx*vx + vy*vy + vz*vz + SV0x_Cr*SV0x_Cr + 2*SV0_Cr*SV0_Cr)
    
    #-----# MF \dot{x}, with "punctual" vLi and (fixed) SigmaVCr
    def MFHam_x(x,y,z,vx,vy,vz,δB):
        as_   = as1461G(δB)
        Rs_   = Rs1461G(δB)
        krel_ = krelMF(vx,vy,vz)
        return -fourπ * as_*as_*as_ * mred_mLi6 * n_Gauss(x,y,z,Natom_Cr,sx_Cr,sy_Cr,sz_Cr) * \
            vx * (1.+ Rs_/as_ * (1.+ krel_*krel_ * Rs_*as_)**2)  \
            / ( (1.+ krel_*krel_ * Rs_*as_)**2 + (krel_*as_)**2 )**2
    
    def MFHam_y(x,y,z,vx,vy,vz,δB):
        as_   = as1461G(δB)
        Rs_   = Rs1461G(δB)
        krel_ = krelMF(vx,vy,vz)
        return -fourπ * as_*as_*as_  * mred_mLi6 * n_Gauss(x,y,z,Natom_Cr,sx_Cr,sy_Cr,sz_Cr) * \
            vy * (1.+ Rs_/as_ * (1.+ krel_*krel_ * Rs_*as_)**2)  \
            / ( (1.+ krel_*krel_ * Rs_*as_)**2 + (krel_*as_)**2 )**2
    
    def MFHam_z(x,y,z,vx,vy,vz,δB):
        as_   = as1461G(δB)
        Rs_   = Rs1461G(δB)
        krel_ = krelMF(vx,vy,vz)
        return -fourπ * as_*as_*as_  * mred_mLi6 * n_Gauss(x,y,z,Natom_Cr,sx_Cr,sy_Cr,sz_Cr) * \
            vz * (1.+ Rs_/as_ * (1.+ krel_*krel_ * Rs_*as_)**2)  \
            / ( (1.+ krel_*krel_ * Rs_*as_)**2 + (krel_*as_)**2 )**2
    
    #-----# MF \dot{v}_x, with "punctual" vLi and (fixed) SigmaVCr.   NB:  divided by m_Li6 with respect to (H10)  [need eq for velocity]
    def MFHam_vx(x,y,z,vx,vy,vz,δB):
        as_   = as1461G(δB)
        Rs_   = Rs1461G(δB)
        krel_ = krelMF(vx,vy,vz)
        return twoπ*as_*hbar_sqr / (mLi6mred*sx_Cr*sx_Cr) * n_Gauss(x,y,z,Natom_Cr,sx_Cr,sy_Cr,sz_Cr) * x * \
               (1.+ krel_*krel_ * Rs_*as_)/((1.+ krel_*krel_ * Rs_*as_)**2 + (krel_*as_)**2)
    
    def MFHam_vy(x,y,z,vx,vy,vz,δB):
        as_   = as1461G(δB)
        Rs_   = Rs1461G(δB)
        krel_ = krelMF(vx,vy,vz)
        return twoπ*as_*hbar_sqr / (mLi6mred*sy_Cr*sy_Cr) * n_Gauss(x,y,z,Natom_Cr,sx_Cr,sy_Cr,sz_Cr) * y * \
               (1.+ krel_*krel_ * Rs_*as_)/((1.+ krel_*krel_ * Rs_*as_)**2 + (krel_*as_)**2)
    
    def MFHam_vz(x,y,z,vx,vy,vz,δB):
        as_   = as1461G(δB)
        Rs_   = Rs1461G(δB)
        krel_ = krelMF(vx,vy,vz)
        return twoπ*as_*hbar_sqr / (mLi6mred*sz_Cr*sz_Cr) * n_Gauss(x,y,z,Natom_Cr,sx_Cr,sy_Cr,sz_Cr) * z * \
               (1.+ krel_*krel_ * Rs_*as_)/((1.+ krel_*krel_ * Rs_*as_)**2 + (krel_*as_)**2)
    
    #------------------------------#





