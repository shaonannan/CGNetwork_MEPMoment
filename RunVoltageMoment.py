import numpy as np
import itertools
import matplotlib.pyplot as plt
import argparse

from internalpopulation import RecurrentPopulation
from externalpopulation import ExternalPopulation
from simulation import Simulation
from connection import Connection as Connection


"""
edited by SYX

initialization  and configuration
vflag for dawson function 
fin for \rho_{Eq}
"""
# python RunVoltageMoment.py -stimulus 1
parser = argparse.ArgumentParser(description='Stimuli & Parameters for the V1 Reduced Model Simulation')

parser.add_argument('-see', type=float, dest='see', default=1.25, help='Weight between short range Excitatory to Excitatory ')
parser.add_argument('-sei', type=float, dest='sei', default=-0.84, help='Weight between short range Inhibitory to Excitatory ')
parser.add_argument('-sie', type=float, dest='sie', default=1.30, help='Weight between short range Inhibitory to Excitatory ')
parser.add_argument('-sii', type=float, dest='sii', default=-0.65, help='Weight between short range Inhibitory to Inhibitory ')
parser.add_argument('-lee', type=float, dest='lee', default=3.745, help='Weight between long range Excitatory to Excitatory ')
parser.add_argument('-lie', type=float, dest='lie', default=6.388, help='Weight between long range Excitatory to Inhibitory')
parser.add_argument('-leef', type=float, dest='leef', default=0.84, help='Weight between long range Excitatory to Excitatory(fast) ')
parser.add_argument('-lief', type=float, dest='lief', default=0.82, help='Weight between long range Excitatory to Inhibitory(fast) ')

parser.add_argument('-stimulus', type=int, dest='stimulus', default=0, help='Different types of stimuli')


args = parser.parse_args()


# intuitive network structure
Net_settings = {'hyp_num': 15,
                'xhyp':5,
                'yhyp':3,
                'xn':4,
                'yn':4,
                'dxx_hyp':500,
                'nmax': 0,
                'Final_time':160,
                'dt':0.1,
                'dv':1e-3}
# here we use orientation and phase
AmpHyp = Net_settings['xn'] * Net_settings['yn'] /2.0
Fun_map_settings = {'ori_num':4,
                   'phase_num':10}
# cell numbers within identical orientation hypercolumn
Cell_type_num = {'e':int(1024/AmpHyp),
                'i':int(1024/AmpHyp)}

ori_2d = np.zeros((Net_settings['xhyp'] * Net_settings['xn'],Net_settings['yhyp'] * Net_settings['yn']))
ori    = np.zeros(Net_settings['hyp_num'] * Net_settings['xn'] * Net_settings['yn'])
interval_ori = 180.0/Fun_map_settings['ori_num']
"""
this part is for generating orientation map (pin-wheel structure)
if np.mod(Net_settings['nx'],2) == 0,upper side cut off = np.floor(Net_settings['nx']/2), cen = cut-off - 0.5 
[0:cut-off] upper side; while [cut-off:] bottom side
if np.mod(Net_settings['nx'],2) == 1, upper side cut off = np.floor((Net_settings['nx']-1)/2), cen = cut-off
[0:cut-off] upper side; maintain unchanged [cut-off]; while bottom side [cut-off+1:]
"""
nbin = Net_settings['xn']
if np.mod(nbin,2) == 0:
    cut_off = np.floor(nbin/2.0)
    cen     = cut_off - 0.5
    (uc,up,bottom) = (int(cut_off),int(cut_off),int(cut_off))
if np.mod(nbin,2) == 1:
    cut_off = np.floor((nbin-1)/2.0)
    cen     = cut_off
    (uc,up,bottom) = (int(cut_off),int(cut_off),int(cut_off+1))
for i in np.arange(1):
    for p in np.arange(Net_settings['xn']):
        for q in np.arange(Net_settings['yn']):
            ttindex = q + p * Net_settings['yn'] + i * Net_settings['yn'] * Net_settings['xn']
            oriabs  = (np.arctan2(p-cen,q-cen)) * 180.0 / np.pi # ttindex
            if oriabs<0.0:
                oriabs = oriabs + 360.0
            oriabs = oriabs * 0.5
            oriabs = np.floor(oriabs/interval_ori)  # for integer not real DEGREE
            if oriabs == Fun_map_settings['ori_num']:
                oriabs = oriabs - 1
            ori[ttindex] = oriabs
            ori_2d[p,q]  = ori[ttindex] # only half(or even smaller) of the total 

if(Net_settings['xhyp'] > 1):
    # 2nd hypercolum! maybe next line~
    # unchanged
    # because of the next line, so we should use x from 'xn' to 2*'xn' y stay in 0:yn
    ori_2d[Net_settings['xn']+uc,:Net_settings['yn']] = ori_2d[uc,:Net_settings['yn']]
    start = 1*Net_settings['xn'] * Net_settings['yn']
    ori_2d[Net_settings['xn']*1:Net_settings['xn']*1+up,:Net_settings['yn']] = ori_2d[Net_settings['xn']-1:bottom-1:-1,:Net_settings['yn']]
    ori_2d[Net_settings['xn']*1+bottom:Net_settings['xn']*2,:Net_settings['yn']] = ori_2d[up-1::-1,:Net_settings['yn']]

    # then here are two templates for 2*k / 2*k+1
    # for 2*k
    ori_template_2k  = np.squeeze(ori_2d[:Net_settings['xn'],:Net_settings['yn']])
    ori_template_2k1 = np.squeeze(ori_2d[Net_settings['xn']:2*Net_settings['xn'],:Net_settings['yn']])
    
    # only take effects on x(long) axis, means being constrainted in the first row
    for ixhyp in range(2,Net_settings['xhyp']):
        if np.mod(ixhyp,2) == 0:
            ori_2d[Net_settings['xn']*ixhyp:Net_settings['xn']*(ixhyp+1),:Net_settings['yn']] = ori_template_2k#np.reshape(ori_template_2k,(3,1))
        if np.mod(ixhyp,2) == 1:
            ori_2d[Net_settings['xn']*ixhyp:Net_settings['xn']*(ixhyp+1),:Net_settings['yn']] = ori_template_2k1#np.reshape(ori_template_2k1,(3,1))

if Net_settings['yhyp']>1:
    # 2nd hypercolum! maybe next column~
    # unchanged
    # because of the next line, so we should use x from 'xn' to 2*'xn' y stay in 0:yn
    ori_2d[:,Net_settings['yn']+uc] = ori_2d[:,uc]
    ori_2d[:,Net_settings['yn']*1:Net_settings['yn']*1+up] = ori_2d[:,Net_settings['yn']-1:bottom-1:-1]
    ori_2d[:,Net_settings['yn']*1+bottom:Net_settings['yn']*2] = ori_2d[:,up-1::-1]

    # then here are two templates for 2*k / 2*k+1
    # for 2*k
    ori_template_2k  = np.squeeze(ori_2d[:,:Net_settings['yn']])
    ori_template_2k1 = np.squeeze(ori_2d[:,Net_settings['yn']:2*Net_settings['yn']])
    
    # only take effects on y(short) axis
    for iyhyp in range(2,Net_settings['yhyp']):
        if np.mod(iyhyp,2) == 0:
            ori_2d[:,Net_settings['yn']*iyhyp:Net_settings['yn']*(iyhyp+1)] = ori_template_2k
        if np.mod(iyhyp,2) == 1:
            ori_2d[:,Net_settings['yn']*iyhyp:Net_settings['yn']*(iyhyp+1)] = ori_template_2k1
ori = np.squeeze(np.reshape(ori_2d,(1,Net_settings['xn']*Net_settings['yn']*Net_settings['hyp_num'])))

''' simple version orientation map'''
'''
ori_interval = np.pi/Net_settings['xn']/Net_settings['yn']
Net_settings['nmax'] = Net_settings['hyp_num'] * Net_settings['xn'] * Net_settings['yn']
ori = np.zeros(Net_settings['nmax'])
for ihyp in range(Net_settings['hyp_num']):
    for ipatch in range(Net_settings['xn'] * Net_settings['yn']):
        idx_tt = ihyp * Net_settings['xn'] * Net_settings['yn'] + ipatch
        ori[idx_tt] = int(ipatch)
'''

def create_functional_columns(Structure_Net,Functional_Map,ori):
    nmax = Structure_Net['hyp_num'] * Structure_Net['xn'] * Structure_Net['yn']
    Structure_Net['nmax'] = nmax

    hypindx = np.arange(Structure_Net['xhyp'])
    hypindy = np.arange(Structure_Net['yhyp'])
    CGindx  = np.arange(Structure_Net['xn'])
    CGindy  = np.arange(Structure_Net['yn'])
    Orientation_map = {}
    Hypercol_map    = {}
    Hypercol        = np.zeros((nmax,3))
    Phase_map       = {}
    index_2_loc     = np.zeros((nmax,3))

    tt_hypx = Structure_Net['yhyp'] * Structure_Net['yn'] * Structure_Net['xn']
    tt_linx = Structure_Net['yhyp'] * Structure_Net['yn']
    tt_lxhy = Structure_Net['yn']
    for ihypx in hypindx:
        for ix in CGindx:
            for ihypy in hypindy:
                for iy in CGindy:
                    ttindex = iy + ihypy * tt_lxhy + ix * tt_linx + ihypx * tt_hypx
                    ttindex_xax = ix + ihypx * Structure_Net['xn']
                    ttindex_yax = iy + ihypy * Structure_Net['yn']
                    # checking ?
                    ttindex_check = ttindex_xax * Structure_Net['yn'] * Structure_Net['yhyp'] + ttindex_yax
                    if (ttindex_check!=ttindex):
                        print('This chessboard-grid is wrong .')
                    #else:
                        #print('Good chessboard-grid.')
                    ttindex_hyp = ihypx * Structure_Net['yhyp'] + ihypy

                    Hypercol_map[(ttindex,'e')] = ttindex_hyp
                    Hypercol_map[(ttindex,'i')] = ttindex_hyp
                    Hypercol[ttindex,2]         = ttindex_hyp
                    Hypercol[ttindex,0]         = ihypx
                    Hypercol[ttindex,1]         = ihypy

                    Phase_map[(ttindex,'e')] = np.random.randint(Fun_map_settings['phase_num'],size = 1)
                    Phase_map[(ttindex,'i')] = np.random.randint(Fun_map_settings['phase_num'],size = 1)
                    Orientation_map[(ttindex,'e')] = ori[ttindex]
                    Orientation_map[(ttindex,'i')] = ori[ttindex]
                    index_2_loc[ttindex,0] = ttindex_hyp
                    index_2_loc[ttindex,1] = ttindex_xax
                    index_2_loc[ttindex,2] = ttindex_yax
    return (Structure_Net,Hypercol,Phase_map,Orientation_map,index_2_loc)

#******************** period distance **********************************************
def preprocess_for_distance_period(index_2_loc,dxx,dyy,Net_settings,Fun_map_settings):
    nmax = Net_settings['nmax']
    # for X distance
    period_x = Net_settings['xn']
    global_x = np.reshape(np.squeeze(index_2_loc[:,1]),(nmax,1))
    global_xmatrix = np.repeat(global_x,nmax,axis=1)
    global_xdis    = np.abs(global_xmatrix - global_xmatrix.T)
    global_xdis    = np.minimum(global_xdis, period_x - global_xdis)**2*dxx
    # for Y distance
    period_y = Net_settings['yn']
    global_y = np.reshape(np.squeeze(index_2_loc[:,2]),(nmax,1))
    global_ymatrix = np.repeat(global_y,nmax,axis=1)
    global_ydis    = np.abs(global_ymatrix - global_ymatrix.T)
    global_ydis    = np.minimum(global_ydis, period_y - global_ydis)**2*dyy
    global_dist    = np.sqrt(global_xdis + global_ydis)
    
    return (global_xdis,global_ydis,global_dist)

# DISTANCE MATRIX !
#******************** cut-off distance **********************************************
def preprocess_for_distance(index_2_loc,dxx,dyy,Net_settings,Fun_map_settings):
    nmax = Net_settings['nmax']
    # for X distance
    global_x = np.reshape(np.squeeze(index_2_loc[:,1]),(nmax,1))
    global_xmatrix = np.repeat(global_x,nmax,axis=1)
    global_xdis    = np.abs(global_xmatrix - global_xmatrix.T)**2*dxx
    # for Y distance
    global_y = np.reshape(np.squeeze(index_2_loc[:,2]),(nmax,1))
    global_ymatrix = np.repeat(global_y,nmax,axis=1)
    global_ydis    = np.abs(global_ymatrix - global_ymatrix.T)**2*dyy
    global_dist    = np.sqrt(global_xdis + global_ydis)
    
    return (global_xdis,global_ydis,global_dist)

def normal_function(x, mean=0, sigma=1.0):
    """
    Returns the value of probability density of normal distribution N(mean,sigma) at point `x`.
    """
    _normalization_factor = np.sqrt(2 * np.pi)

#    return np.exp(-np.power((x - mean)/sigma, 2)/2) / (sigma * _normalization_factor)
    return np.exp(-np.power((x - mean)/sigma, 2)/1.0) / (sigma * _normalization_factor) # do not divide 2.0

def circular_dist(a, b, period):
    """
    Returns the distance between a and b (scalars) in a domain with `period` period.
    """
    return np.minimum(np.abs(a - b), period - np.abs(a - b))

# Create visual stimuli
def utilhvsvec(x):
    f = np.maximum(x,0)
    return f

def visual_temporal_kernel(t1,t2,dt,tfinal):
    tt = np.arange(0,tfinal,dt)
    f  = tt/(t1**2)*np.exp(-tt/t1) - tt/(t2**2)*np.exp(-tt/t2)
    return f

def input_convolution(t_pulse,dt,tfinal,nparam,tparam):
    ntt = int(tfinal/dt)
    npp = int(t_pulse/dt)
    square_pulse = np.ones(npp)
    t_resp       = np.zeros((nparam,ntt))
    if args.stimulus < 1:
        # number of different temporal kernels
        for itk in range(nparam):
            t1 = tparam[itk,0]
            t2 = tparam[itk,1]
            tkernel = visual_temporal_kernel(t1,t2,dt,tfinal+5)
            t_k     = np.convolve(tkernel,square_pulse*dt,mode = 'full')
            t_resp[itk,:] = t_k[:ntt]
            # normalize
            t_resp[itk,:] = t_resp[itk,:] / np.max(np.squeeze(t_resp[itk,:]))
            t_resp[itk,:] = utilhvsvec(t_resp[itk,:])         
    else:    
        # number of different temporal kernels
        for itk in range(nparam):
            t1 = tparam[itk,0]
            t2 = tparam[itk,1]
            if itk<1 :
                square_bright = np.ones(npp*3)
                tkernel = visual_temporal_kernel(t1,t2,dt,tfinal+5)
                t_k     = np.convolve(tkernel,square_bright*dt,mode = 'full')  
            else:
                tkernel = visual_temporal_kernel(t1,t2,dt,tfinal+5)
                t_k     = np.convolve(tkernel,square_pulse*dt,mode = 'full')
            t_resp[itk,:] = t_k[:ntt]
            # normalize
            t_resp[itk,:] = t_resp[itk,:] / np.max(np.squeeze(t_resp[itk,:]))
            t_resp[itk,:] = utilhvsvec(t_resp[itk,:])

    t_brightened = t_resp[0,:] - 0.65*t_resp[1,:]
    t_brightened = t_brightened/np.max(t_brightened)
    t_brightened*= 0.32
    t_darken     = t_resp[1,:] - 0.65*t_resp[0,:]
    t_darken     = t_darken/np.max(t_darken)
    t_darken    *= 0.32
    # heaviside function
    t_brightened = utilhvsvec(t_brightened)
    t_darken     = utilhvsvec(t_darken)
    return t_resp,t_brightened,t_darken

Net_settings,Hypm,Pham,Orim,index_2_loc = create_functional_columns(Net_settings,Fun_map_settings,ori)            
Net_settings['nmax'] = Net_settings['xn'] * Net_settings['yn'] * Net_settings['hyp_num']
# MODIFYING 'nmax' in Net_settings, which is used in latter code

"""
using in real python code 2018/04/02 version 0
"""
# Simulation settings:
t0 = 0.0
dt = Net_settings['dt']
tf = Net_settings['Final_time']
dv = Net_settings['dv']
verbose = True

update_method = 'approx'
approx_order = 1
tol  = 1e-14

(dt, tfinal, t_pulse) = (dt, tf, 68)
nparam = 2
# (t1on,t2on,t1off,t2off) = (0.014/0.056*0.036,0.036,0.014/0.056*0.036,0.036)#(0.014,0.056,0.014/0.056*0.036,0.036)
tparam = np.zeros((nparam,2))
tparam[0,0] = 14.0
tparam[0,1] = 56.0
tparam[1,0] = 14.0/56.0*36.0
tparam[1,1] = 36.0

active_resp,active_brightened,active_darken = input_convolution(t_pulse,dt,tfinal,nparam,tparam)
# here, not only we use specific locations to receive visual stimuli, but also particular stimuli are chosen to 
# be assignedlike assigning brightened and darken stimuli respectively.

stimuli_collection = {}
tt_delay = 10.0
tmp_brightened = np.zeros_like(active_brightened) 
tmp_darken     = np.zeros_like(active_darken) 
nnn = len(active_brightened)
nnt_delay = int(tt_delay/dt)

tmp_brightened[nnt_delay:] = active_brightened[:(nnn-nnt_delay)] 
active_brightened = tmp_brightened
tmp_darken[nnt_delay:] = active_darken[:(nnn-nnt_delay)] 
active_darken = (tmp_darken/0.3)*0.90


if args.stimulus==0:
    # artificial time delay
    t_delay_brightened = 0.0
    tmp_brightened = np.zeros_like(active_brightened) 
    nnn = len(active_brightened)
    nnt_delay = int(t_delay_brightened/dt)
    tmp_brightened[nnt_delay:] = active_brightened[:(nnn-nnt_delay)] 
    active_brightened = 1.00 * tmp_brightened
    
    stm_region = np.zeros((Net_settings['xn'] *Net_settings['xhyp'] ,Net_settings['yn'] *Net_settings['yhyp']))
    stm_cue = np.zeros_like(stm_region)
    
    stm_region[11:17,3:4*2+1] = 1
    stm_cue[4:2*4+2,3:4*2+1]    = 1
    '''
    # plotting stimulus
    plt.figure()
    plt.subplot(2,1,1)
    plt.imshow(stm_region)
    plt.subplot(2,1,2)
    plt.imshow(stm_cue)
    '''
    stm_region = np.reshape(stm_region,(4*5*4*3,1))
    stm_cue    = np.reshape(stm_cue,(4*5*4*3,1))
    
    id_stm = np.where(stm_region)
    id_cue = np.where(stm_cue)
    
    stimuli_collection['brightened'] = active_brightened
    stimuli_collection['darken']     = active_darken
    
    tt_dark = 6*6  # 2 Hypercolumns would receive cue stimulus (flashed dark square)
    tt_bright = 6*6  #Net_settings['nmax'] - tt_dark             # remaining Hypercolumns would receive long stationary bar
    
    stimuli_type =['darken']
    for i in range(1,tt_dark):
        stimuli_type.append('darken')
    for i in range(tt_bright):
        stimuli_type.append('brightened')
    
    # >>> activation location
    active_location = [0]
    for i in range(6*6):# dark
        idloc = id_cue[0][i]
        active_location.append(idloc)
    
    for i in range(6*6):# bright
        idloc = id_stm[0][i]
        active_location.append(idloc)
    
    active_type = ['b']
    for i in range(1,12*6):#1,Net_settings['nmax']):
        active_type.append('b')

if args.stimulus==1:  
    # artificial time delay
    t_delay_brightened = 30.0
    tmp_brightened = np.zeros_like(active_brightened) 
    nnn = len(active_brightened)
    nnt_delay = int(t_delay_brightened/dt)
    tmp_brightened[nnt_delay:] = active_brightened[:(nnn-nnt_delay)] 
    active_brightened = 1.00 * tmp_brightened
    
    stm_region = np.zeros((Net_settings['xn'] *Net_settings['xhyp'] ,Net_settings['yn'] *Net_settings['yhyp']))
    stm_cue = np.zeros_like(stm_region)

    stm_region[3:4 * 4 + 1, 3:4 * 2 + 1] = 1
    stm_cue[3:2 * 4 + 1, 3:4 * 2 + 1] = 1
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(stm_region)
    plt.subplot(2, 1, 2)
    plt.imshow(stm_cue)
    stm_region = np.reshape(stm_region, (4 * 5 * 4 * 3, 1))
    stm_cue = np.reshape(stm_cue, (4 * 5 * 4 * 3, 1))
    
    id_stm = np.where(stm_region)
    id_cue = np.where(stm_cue)
    
    stimuli_collection['brightened'] = active_brightened
    stimuli_collection['darken'] = active_darken
    
    tt_dark = 6 * 6  # 2 Hypercolumns would receive cue stimulus (flashed dark square)
    tt_bright = 8 * 6  # Net_settings['nmax'] - tt_dark             # remaining Hypercolumns would receive long stationary bar
    
    stimuli_type = ['darken']
    for i in range(1, tt_dark):
        stimuli_type.append('darken')
    for i in range(tt_bright):
        stimuli_type.append('brightened')
    
    active_location = [0]
    for i in range(14 * 6):  # Net_settings['nmax']):
        idloc = id_stm[0][i]
        active_location.append(idloc)
    
    active_type = ['b']
    for i in range(1, 14 * 6):  # 1,Net_settings['nmax']):
        active_type.append('b')

def input_signal_stream(active_location, active_type, active_resp,t_properties,Net_settings):
    if args.stimulus ==0:
        ''' id_cue '''
        stm_cue = np.zeros((Net_settings['xn'] *Net_settings['xhyp'] ,Net_settings['yn'] *Net_settings['yhyp']))
        stm_cue[4:2*4+2,3:4*2+1]     = 1
        plt.figure()
        plt.imshow(stm_cue)
        stm_cue    = np.reshape(stm_cue,(4*5*4*3,1))
        ''' end for id_cue '''
    if args.stimulus == 1:
        ''' id_cue !!! '''
        stm_cue = np.zeros((Net_settings['xn'] *Net_settings['xhyp'] ,Net_settings['yn'] *Net_settings['yhyp']))
        stm_cue[3:2*4+1,3:4*2+1]    = 1
        plt.figure()
        plt.imshow(stm_cue)
        stm_cue    = np.reshape(stm_cue,(4*5*4*3,1))
        ''' end for id_cue '''        
    
    (dt, tfinal) = t_properties
    ntt = int(tfinal/dt)
    # Create base-visual stimuli
    External_stimuli_dict = np.ones(ntt) * 0.07
    # Create populations:
    background_population_dict = {}
    internal_population_dict = {}
    CG_range = np.arange(Net_settings['nmax'])

    # base activities
    for layer, celltype in itertools.product(CG_range, ['e', 'i']):  
        background_population_dict[layer,celltype] = ExternalPopulation(External_stimuli_dict,dt, record=False)
        internal_population_dict[layer, celltype]  = RecurrentPopulation(dt = dt,v_min=-1.0, v_max=1.0, dv=dv, update_method=update_method,
                                approx_order=approx_order, tol=tol, hyp_idx = 0,ei_pop = celltype,NumCell = Cell_type_num[celltype])
    # choose mode, hypercolumn or subpopulation
    mode = active_location[0]
    if(mode ==  0): # subpopulation
        print('line 564 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('Total actived regions: ',len(active_type))
        for iactive in range(len(active_type)):
            stm_type = stimuli_type[iactive]
            # particular subpopulation (not hypercolumn)
            isubpopulation = active_location[iactive+1]
            iloc = isubpopulation
            
            # cell type ? 'b' or 'e'/'i'
            celltype = active_type[iactive]
            if(celltype == 'b'):
                if(stm_cue[iloc,0] ==1 ):
                    print('cue location: ',iloc)
                    if args.stimulus < 1:
                        particular_stimulus = 0.0*stimuli_collection['brightened']+1.0*stimuli_collection['darken']
                    else:
                        particular_stimulus = 1.0*stimuli_collection['brightened']+1.0*stimuli_collection['darken']
                else:
                    if(stm_type == 'brightened'):
                        particular_stimulus = stimuli_collection['brightened']#np.zeros_like(stimuli_collection[stm_type])
                    else:
                        particular_stimulus = np.zeros_like(stimuli_collection[stm_type])
                temp_inp= External_stimuli_dict[:ntt] + np.squeeze(particular_stimulus[:ntt])
                background_population_dict[iloc,'e'] = ExternalPopulation(temp_inp,dt,record=False)
                background_population_dict[iloc,'i'] = ExternalPopulation(temp_inp,dt,record=False)
                        
            else:
                if(id_cue[iloc,0] ==1 ):
                    particular_stimulus = stimuli_collection['brightened']+stimuli_collection['darken']
                else:
                    particular_stimulus = stimuli_collection[stm_type]
                temp_inp = External_stimuli_dict[:ntt] + np.squeeze(particular_stimulus[:ntt])
                background_population_dict[iloc,celltype] = ExternalPopulation(temp_inp,dt,record=False)

    if(mode == -1): # hypercolumn
        for iactive in range(len(active_type)):
            # no matter which particular location is target population,  the characteristics of
            # particular stimuli should be determined already!

            # particular stimulus
            stm_type = stimuli_type[iactive]
            # particular target regions
            ihyper = active_location[iactive+1]
            ilocst = ihyper * Net_settings['xn'] * Net_settings['yn']
            iloced = (ihyper+1) * Net_settings['xn'] * Net_settings['yn']
            iloc_range = np.arange(ilocst,iloced)

            # cell types ? 'b' or 'e/i'
            celltype = active_type[iactive]
            if (celltype == 'b'):
                particular_stimulus = stimuli_collection[stm_type]
                temp_inp = External_stimuli_dict[:ntt] + np.squeeze(particular_stimulus[:ntt])
                for ilayer, itype in itertools.product(iloc_range, ['e', 'i']):
                    # both excitatory and inhibitory
                    background_population_dict[ilayer,itype] = ExternalPopulation(temp_inp,dt,record=False)

            else:
                particular_stimulus = stimuli_collection[stm_type]
                temp_inp = External_stimuli_dict[:ntt] + np.squeeze(particular_stimulus[:ntt])
                for ilayer, itype in itertools.product(iloc_range, celltype):
                    # only subpopulation
                    background_population_dict[ilayer,itype] = ExternalPopulation(temp_inp,dt,record=False)
        
    population_list = list(background_population_dict.values()) + list(internal_population_dict.values())
    return (background_population_dict,internal_population_dict,population_list)

background_population_dict,internal_population_dict,population_list = input_signal_stream(active_location,active_type,active_resp,[dt,tfinal],Net_settings)
    
# PARAMETERS FOR CONNECTIVITY
(denexc,deninh,axnexc,axninh) = (50.0,50.0,200.0,100.0)
sr_exc2exc = np.sqrt(denexc * denexc + axnexc * axnexc)
sr_inh2exc = np.sqrt(denexc * denexc + axninh * axninh)
sr_exc2inh = np.sqrt(deninh * deninh + axnexc * axnexc)
sr_inh2inh = np.sqrt(deninh * deninh + axninh * axninh)

sr_sigma = {'short_range_exc_exc':sr_exc2exc,
           'short_range_exc_inh':sr_exc2inh,
           'short_range_inh_exc':sr_inh2exc,
           'short_range_inh_inh':sr_inh2inh} # source2target
lr_sigma = {'long_range_exc_exc': 1.00*1000,
            'long_range_exc_inh': 1.00*1000}
NE_source = Cell_type_num['e']
NI_source = Cell_type_num['i']
# >>>>>>>>>>>>>>>>>>> About 'NORMALIZATION' >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.
AmpHyp = Net_settings['xn'] * Net_settings['yn'] /2.0   # 2 is original CG patches in identical orientation hypercolumn
AmpOri = Fun_map_settings['ori_num'] / 2.0
g_norm = {'short_range_exc_exc':args.see/NE_source/Net_settings['xn']/Net_settings['yn'],#0.24/NE_source,
           'short_range_exc_inh':args.sie/NE_source/Net_settings['xn']/Net_settings['yn'],#0.33/NE_source,
           'short_range_inh_exc':args.sei/NI_source/Net_settings['xn']/Net_settings['yn'],#-0.44/NI_source,
           'short_range_inh_inh':args.sii/NI_source/Net_settings['xn']/Net_settings['yn'],#-0.09/NI_source,
           
           'long_range_exc_exc_fast':args.leef/NE_source/AmpHyp * AmpOri,#0.364/NE_source,#0.426/NE_source,#0.24/NE_source,
           'long_range_exc_inh_fast':args.lief/NE_source/AmpHyp * AmpOri,##0.396/NE_source,#0.33/NE_source,
           
           'long_range_exc_exc':args.lee/NE_source/AmpHyp * AmpOri,#2.46,#1.64,#2.56,#2.46,#1.26,#1.28,
           'long_range_exc_inh':args.lie/NE_source/AmpHyp * AmpOri}#2.86}#1.96}#2.86}#2.86}#1.64}#1.96}

# FUNCTIONS FOR CONNECTIVITY
def cortical_to_cortical_connection_normalization(global_dist,sr_sigma,lr_sigma,ori_map,hyp_map_tt,Net_settings,Fun_map_settings):
    # for short-range connections normalization
    norm_cortical_connection = {}
    nmax    = Net_settings['nmax']
    # for long range connextions in different hypercolumns
    nori = Fun_map_settings['ori_num']
    # initialize
    norm_cortical_connection['lr','exc2exc'] = np.zeros((nmax,nmax))
    norm_cortical_connection['lr','exc2inh'] = np.zeros((nmax,nmax))
    """
    """
    # the 1st algorithm, using old code  hyp_map == np.squeeze(hyp_map_tt[:,2])
    # the second algorithm, using identical long-range connections hypx_map = np.squeeze(hyp_map_tt[:,0])
    # hypy_map = np.squeeze(hyp_map_tt[:,1])
    hyp_map = np.squeeze(hyp_map_tt[:,2])
    hypx_map = np.squeeze(hyp_map_tt[:,0])
    hypy_map = np.squeeze(hyp_map_tt[:,1])
    # total normalized value
    lrhyp_xdis,lrhyp_ydis,lrhyp_dist,cen_amp = create_longrange_connection(Net_settings,lr_sigma)

#    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> the 1st algorithm >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#    for ihyp in range(Net_settings['hyp_num']): # within identical orientation hyper-column
#        for iori in range(nori):
#            index_hyper = np.where(hyp_map == ihyp)
#            index_orien = np.where(ori_map == iori)
#            index_target = np.intersect1d(index_hyper,index_orien)
#            index_source = np.setdiff1d(index_orien,index_hyper)
#            ee_curr = np.zeros((nmax,nmax))
#            
#            [x,y] = np.meshgrid(index_target,index_source)
#            
#            xtarget,xsource = hypx_map[index_target],hypx_map[index_source]
#            ytarget,ysource = hypy_map[index_target],hypy_map[index_source]
#            
#            [xhypx,xhypy] = np.meshgrid(xtarget,xsource)
#            xhyp_dist = np.abs(xhypx-xhypy)
#            xhyp_dist= xhyp_dist.astype(int)
#            
#            [yhypx,yhypy] = np.meshgrid(ytarget,ysource)
#            yhyp_dist = np.abs(yhypx-yhypy)
#            yhyp_dist= yhyp_dist.astype(int)
#            
#            ee_cen = cen_amp['lr','exc2exc']
#            ee_curr[x.T,y.T] = ee_cen[xhyp_dist.T,yhyp_dist.T]
#            
#            norm_cortical_connection['lr','exc2exc'] = norm_cortical_connection['lr','exc2exc'] + ee_curr
#            
#            ei_curr = np.zeros((nmax,nmax))
#            [x,y] = np.meshgrid(index_target,index_source)
#            
#            xtarget,xsource = hypx_map[index_target],hypx_map[index_source]
#            ytarget,ysource = hypy_map[index_target],hypy_map[index_source]
#            
#            [xhypx,xhypy] = np.meshgrid(xtarget,xsource)
#            xhyp_dist = np.abs(xhypx-xhypy)
#            xhyp_dist= xhyp_dist.astype(int)
#            
#            [yhypx,yhypy] = np.meshgrid(ytarget,ysource)
#            yhyp_dist = np.abs(yhypx-yhypy)
#            yhyp_dist= yhyp_dist.astype(int)
#            
#            ei_cen = cen_amp['lr','exc2inh']
#            ei_curr[x.T,y.T] = ei_cen[xhyp_dist.T,yhyp_dist.T]
#            norm_cortical_connection['lr','exc2inh'] = norm_cortical_connection['lr','exc2inh'] + ei_curr

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>> the 2nd Algorithm >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    for ihyp in range(Net_settings['hyp_num']): # within identical orientation hyper-column
        for iori in range(nori):
            index_hyper = np.where(hyp_map == ihyp)
            index_orien = np.where(ori_map == iori)
            index_target = np.intersect1d(index_hyper,index_orien)
            index_source = np.setdiff1d(index_orien,index_hyper)
            ee_curr = np.zeros((nmax,nmax))
            lentarget = len(index_target)
            lensource = len(index_source)
            [x,y] = np.meshgrid(index_target,index_source)
            
            A_temp  = (np.squeeze(global_dist[x,y])).T
            A_temp  = np.reshape(A_temp,(lentarget,lensource))
            
            A_temp  = normal_function(A_temp,0,lr_sigma['long_range_exc_exc'])
            A_sum   = np.reshape(np.sum(A_temp,axis = 1),(lentarget,1))
            A_sum   = np.ones_like(A_sum)/A_sum
            A_sum   = np.repeat(A_sum,lensource,axis = 1)
            
            ee_curr[x.T,y.T] = A_temp * A_sum 
            norm_cortical_connection['lr','exc2exc'] = norm_cortical_connection['lr','exc2exc'] + ee_curr
            
            ei_curr = np.zeros((nmax,nmax))
            lentarget = len(index_target)
            lensource = len(index_source)
            [x,y] = np.meshgrid(index_target,index_source)
            
            A_temp  = (np.squeeze(global_dist[x,y])).T
            A_temp  = np.reshape(A_temp,(lentarget,lensource))
            
            A_temp  = normal_function(A_temp,0,lr_sigma['long_range_exc_inh'])
            A_sum   = np.reshape(np.sum(A_temp,axis = 1),(lentarget,1))
            A_sum   = 1.0/np.repeat(A_sum,lensource,axis = 1)
            
            ei_curr[x.T,y.T] = A_temp * A_sum 
            norm_cortical_connection['lr','exc2inh'] = norm_cortical_connection['lr','exc2inh'] + ei_curr
            
    # for short range connextions within identical hypercolumn
    nmax    = Net_settings['nmax']
    nmaxhyp = Net_settings['xn'] * Net_settings['yn']
    
    norm_cortical_connection['sr','exc2exc'] = np.zeros((nmax,nmax))
    norm_cortical_connection['sr','exc2inh'] = np.zeros((nmax,nmax))
    norm_cortical_connection['sr','inh2exc'] = np.zeros((nmax,nmax))
    norm_cortical_connection['sr','inh2inh'] = np.zeros((nmax,nmax))
    for ihyp in range(Net_settings['hyp_num']):
        index_start = nmaxhyp * ihyp 
        index_end   = nmaxhyp * (ihyp + 1)
        
        # exc to exc
        ee_curr = np.zeros((nmax,nmax))
        A_temp  = np.squeeze(global_dist[index_start:index_end,:])
        A_temp  = normal_function(A_temp,0,sr_sigma['short_range_exc_exc'])
        A_sum   = np.reshape(np.sum(A_temp,axis = 1),(nmaxhyp,1))
        A_sum   = 1.0/np.repeat(A_sum,nmax,axis = 1)
        A_norm  = (A_temp * A_sum) # x - target cell y source cell
        ee_curr[index_start:index_end,:] = A_norm
        norm_cortical_connection['sr','exc2exc'] = norm_cortical_connection['sr','exc2exc'] + ee_curr
        
        # exc to inh
        ei_curr = np.zeros((nmax,nmax))
        A_temp  = np.squeeze(global_dist[index_start:index_end,:])
        A_temp  = normal_function(A_temp,0,sr_sigma['short_range_exc_inh'])
        A_sum   = np.reshape(np.sum(A_temp,axis = 1),(nmaxhyp,1))
        A_sum   = 1.0/np.repeat(A_sum,nmax,axis = 1)
        A_norm  = (A_temp * A_sum) # x - target cell y source cell
        ei_curr[index_start:index_end,:] = A_norm
        norm_cortical_connection['sr','exc2inh'] = norm_cortical_connection['sr','exc2inh'] + ei_curr
        
        # inh to exc
        ie_curr = np.zeros((nmax,nmax))
        A_temp  = np.squeeze(global_dist[index_start:index_end,:])
        A_temp  = normal_function(A_temp,0,sr_sigma['short_range_inh_exc'])
        A_sum   = np.reshape(np.sum(A_temp,axis = 1),(nmaxhyp,1))
        A_sum   = 1.0/np.repeat(A_sum,nmax,axis = 1)
        A_norm  = (A_temp * A_sum) # x - target cell y source cell
        ie_curr[index_start:index_end,:] = A_norm
        norm_cortical_connection['sr','inh2exc'] = norm_cortical_connection['sr','inh2exc'] + ie_curr
        
        # inh to inh
        ii_curr = np.zeros((nmax,nmax))
        A_temp  = np.squeeze(global_dist[index_start:index_end,:])
        A_temp  = normal_function(A_temp,1,sr_sigma['short_range_inh_inh'])
        A_sum   = np.reshape(np.sum(A_temp,axis = 1),(nmaxhyp,1))
        A_sum   = 1.0/np.repeat(A_sum,nmax,axis = 1)
        A_norm  = (A_temp * A_sum) # x - target cell y source cell
        ii_curr[index_start:index_end,:] = A_norm
        norm_cortical_connection['sr','inh2inh'] = norm_cortical_connection['sr','inh2inh'] + ii_curr
        
    return norm_cortical_connection
    
def cortical_to_cortical_connection(background_population_dict, internal_population_dict, g_norm, delay, orientations,
                                    phases,cortical_norm):
    connection_list = []
    """
    Could be used to create DEE/DIE/DEI/DII/(DEE with LEEf/DIE with LIEf)
    """
    nmax = Net_settings['nmax']
    DEE,DEI,DIE,DII = np.zeros((nmax,nmax)),np.zeros((nmax,nmax)),np.zeros((nmax,nmax)),np.zeros((nmax,nmax))

    # feedforward connections
    for ihyp in range(Net_settings['hyp_num']):
        num_in_hyp = Net_settings['xn'] * Net_settings['yn']
        start_id = ihyp * num_in_hyp
        end_id   = (ihyp + 1) * num_in_hyp
        
        for ixs in range(Net_settings['xn']):
            for iys in range(Net_settings['yn']):
                sub_index_seed = iys + ixs * ( Net_settings['yn'])
                g_index_seed   = sub_index_seed + start_id
                
                # source_exc, target_exc
                source_population = background_population_dict[g_index_seed,'e']
                target_population = internal_population_dict[g_index_seed,'e']
                curr_connection = Connection(source_population,target_population,nsyn = 1.0,nsyn_post = Cell_type_num['e'], 
                                             weights = 0.132,probs = 1.0,conn_type = 'ShortRange',v_min = -1.0,v_max = 1.0,dv = dv)#0.030#
                connection_list.append(curr_connection)
                
                # source_inh, target_inh
                source_population = background_population_dict[g_index_seed,'i']
                target_population = internal_population_dict[g_index_seed,'i']
                curr_connection = Connection(source_population,target_population,nsyn = 1.0,nsyn_post = Cell_type_num['i'], 
                                             weights = 0.1314,probs = 1.0,conn_type = 'ShortRange',v_min = -1.0,v_max = 1.0,dv = dv)#0.0290#
                connection_list.append(curr_connection)
    """
    """
    # short-range connectivity nomalized 
    ge2e    = g_norm['short_range_exc_exc']
    ge2i    = g_norm['short_range_exc_inh']
    gi2e    = g_norm['short_range_inh_exc']
    gi2i    = g_norm['short_range_inh_inh']  
    
    gle2ef   = g_norm['long_range_exc_exc_fast']
    gle2if   = g_norm['long_range_exc_inh_fast']   
    
    gle2e   = g_norm['long_range_exc_exc']
    gle2i   = g_norm['long_range_exc_inh']              

    
    short_range_exc2exc = cortical_norm['sr','exc2exc']
    short_range_exc2inh = cortical_norm['sr','exc2inh']
    short_range_inh2exc = cortical_norm['sr','inh2exc']
    short_range_inh2inh = cortical_norm['sr','inh2inh']
    
    long_range_exc2exc  = cortical_norm['lr','exc2exc']
    long_range_exc2inh  = cortical_norm['lr','exc2inh']
    """
    """
    # self-connectivity (only when g_index_seed == target_index)
    for ihyp in range(Net_settings['hyp_num']):
        num_in_hyp = Net_settings['xn'] * Net_settings['yn']
        start_id = ihyp * num_in_hyp
        end_id   = (ihyp + 1) * num_in_hyp
        
        for ixs in range(Net_settings['xn']):
            for iys in range(Net_settings['yn']):
                sub_index_seed = iys + ixs * ( Net_settings['yn'])
                g_index_seed   = sub_index_seed + start_id
                i_up_triangle  = g_index_seed

                # source_exc, target_exc
                if(short_range_exc2exc[i_up_triangle,g_index_seed]>(4.7*1e-6)):
                    source_population = internal_population_dict[g_index_seed,'e']
                    target_population = internal_population_dict[i_up_triangle,'e']
                    curr_connection = Connection(source_population,target_population,nsyn = Cell_type_num['e'],nsyn_post = Cell_type_num['e'], 
                                             weights = ge2e * short_range_exc2exc[i_up_triangle,g_index_seed],probs = 1.0,conn_type = 'ShortRange',v_min = -1.0,v_max = 1.0,dv = dv)

                    DEE[i_up_triangle,g_index_seed] += ge2e * short_range_exc2exc[i_up_triangle,g_index_seed]

                    connection_list.append(curr_connection)

                # source_exc, target_inh
                if(short_range_exc2inh[i_up_triangle,g_index_seed]>(4.7*1e-6)):
                    source_population = internal_population_dict[g_index_seed,'e']
                    target_population = internal_population_dict[i_up_triangle,'i']
                    curr_connection = Connection(source_population,target_population,nsyn = Cell_type_num['e'],nsyn_post = Cell_type_num['i'], 
                                             weights = ge2i * short_range_exc2inh[i_up_triangle,g_index_seed],probs = 1.0,conn_type = 'ShortRange',v_min = -1.0,v_max = 1.0,dv = dv)

                    DIE[i_up_triangle,g_index_seed] += ge2i  * short_range_exc2inh[i_up_triangle,g_index_seed]

                    connection_list.append(curr_connection)

                # source_inh, target_exc
                if(short_range_inh2exc[i_up_triangle,g_index_seed]>(4.7*1e-6)):
                    source_population = internal_population_dict[g_index_seed,'i']
                    target_population = internal_population_dict[i_up_triangle,'e']
                    curr_connection = Connection(source_population,target_population,nsyn = Cell_type_num['i'],nsyn_post = Cell_type_num['e'], 
                                             weights = gi2e * short_range_inh2exc[i_up_triangle,g_index_seed],probs = 1.0,conn_type = 'ShortRange',v_min = -1.0,v_max = 1.0,dv = dv)

                    DEI[i_up_triangle,g_index_seed] += gi2e  * short_range_inh2exc[i_up_triangle,g_index_seed]

                    connection_list.append(curr_connection)

                # source_inh, target_inh
                if(short_range_inh2inh[i_up_triangle,g_index_seed]>(4.7*1e-6)):
                    source_population = internal_population_dict[g_index_seed,'i']
                    target_population = internal_population_dict[i_up_triangle,'i']
                    curr_connection = Connection(source_population,target_population,nsyn = Cell_type_num['i'],nsyn_post = Cell_type_num['i'], 
                                             weights = gi2i * short_range_inh2inh[i_up_triangle,g_index_seed],probs = 1.0,conn_type = 'ShortRange',v_min = -1.0,v_max = 1.0,dv = dv)

                    DII[i_up_triangle,g_index_seed] += gi2i  * short_range_inh2inh[i_up_triangle,g_index_seed]

                    connection_list.append(curr_connection)
    """
    """
    # long-range connectivity 
    # choose excitatory as !!! source neuron !!!
    for ihyp in range(Net_settings['hyp_num']):
        num_in_hyp = Net_settings['xn'] * Net_settings['yn']
        start_id = ihyp * num_in_hyp
        end_id   = (ihyp + 1) * num_in_hyp
        
        for ixs in range(Net_settings['xn']):
            for iys in range(Net_settings['yn']):
                sub_index_seed = iys + ixs * ( Net_settings['yn'])
                g_index_seed   = sub_index_seed + start_id
                
                long_range_exc2exc  = cortical_norm['lr','exc2exc']
                long_range_exc2inh  = cortical_norm['lr','exc2inh']
                
                # long-range excitatory 2 excitatory
                # already know seed cell index
                # long-range connectivity [target,source]
                # choose excitatory subpopulation as source neuron
                total_exc2exc = np.squeeze(long_range_exc2exc[:,g_index_seed])
                total_exc2inh = np.squeeze(long_range_exc2inh[:,g_index_seed])
                # effective targetneuron
                effective_target_exc = np.where(total_exc2exc)
                effective_target_inh = np.where(total_exc2inh)
                
                for itarget in range(len(effective_target_exc[0])):
                    icell = effective_target_exc[0][itarget]
                    checkicell = effective_target_inh[0][itarget]
                    if(icell!=checkicell):
                        print('Cell-index for Long-range connectivity is wrong!\n')
                    if(long_range_exc2exc[icell,g_index_seed]>(4.7*1e-6)):
                        source_population = internal_population_dict[g_index_seed,'e']
                        # excitatory target
                        target_population = internal_population_dict[icell,'e']
                        curr_connection = Connection(source_population,target_population,nsyn = Cell_type_num['e'],nsyn_post = Cell_type_num['e'],
                                             weights = gle2e * long_range_exc2exc[icell,g_index_seed],probs = 1.0,conn_type = 'LongRange',v_min = -1.0,v_max = 1.0,dv = dv)

                        connection_list.append(curr_connection)
                        
                        # long-range fast synaptic input
                        curr_connection = Connection(source_population,target_population,nsyn = Cell_type_num['e'],nsyn_post = Cell_type_num['e'],
                                             weights = gle2ef * long_range_exc2exc[icell,g_index_seed],probs = 1.0,conn_type = 'ShortRange',v_min = -1.0,v_max = 1.0,dv = dv)

                        DEE[icell,g_index_seed] += gle2ef * long_range_exc2exc[icell,g_index_seed]

                        connection_list.append(curr_connection)
                        
                    # inhibitory target
                    if(long_range_exc2inh[checkicell,g_index_seed]>(4.7*1e-6)):
                        source_population = internal_population_dict[g_index_seed,'e']
                        target_population = internal_population_dict[checkicell,'i']
                        curr_connection = Connection(source_population,target_population,nsyn = Cell_type_num['e'],nsyn_post = Cell_type_num['i'], 
                                             weights = gle2i * long_range_exc2inh[checkicell,g_index_seed],probs = 1.0,conn_type = 'LongRange',v_min = -1.0,v_max = 1.0,dv = dv)

                        connection_list.append(curr_connection)
                        
                        # long-range fast synaptic input
                        curr_connection = Connection(source_population,target_population,nsyn = Cell_type_num['e'],nsyn_post = Cell_type_num['i'], 
                                             weights = gle2if * long_range_exc2inh[checkicell,g_index_seed],probs = 1.0,conn_type = 'ShortRange',v_min = -1.0,v_max = 1.0,dv = dv)

                        DIE[checkicell,g_index_seed] += gle2if * long_range_exc2inh[checkicell,g_index_seed]

                        connection_list.append(curr_connection)
    """
    """
    nmax = Net_settings['hyp_num'] * Net_settings['xn'] * Net_settings['yn']
    # short-range connectivity, different patches, not 'delta' self-connectivity
    for ihyp in range(Net_settings['hyp_num']):
        num_in_hyp = Net_settings['xn'] * Net_settings['yn']
        start_id = ihyp * num_in_hyp
        end_id   = (ihyp + 1) * num_in_hyp
        
        for ixs in range(Net_settings['xn']):
            for iys in range(Net_settings['yn']):
                sub_index_seed = iys + ixs * ( Net_settings['yn'])
                g_index_seed   = sub_index_seed + start_id
                for i_up_triangle in range(g_index_seed+1,nmax):#end_id):

                    # source_exc, target_exc
                    if(short_range_exc2exc[i_up_triangle,g_index_seed]>(4.7*1e-6)):
                        source_population = internal_population_dict[g_index_seed,'e']
                        target_population = internal_population_dict[i_up_triangle,'e']

                        curr_connection = Connection(source_population,target_population,nsyn = Cell_type_num['e'],nsyn_post = Cell_type_num['e'],
                                             weights = ge2e * short_range_exc2exc[i_up_triangle,g_index_seed],probs = 1.0,conn_type = 'ShortRange',v_min = -1.0,v_max = 1.0,dv = dv)

                        DEE[i_up_triangle,g_index_seed] += ge2e * short_range_exc2exc[i_up_triangle,g_index_seed]

                        connection_list.append(curr_connection)
                    
                    # inverse
                    if(short_range_exc2exc[g_index_seed,i_up_triangle]>(4.7*1e-6)):
                        source_population = internal_population_dict[i_up_triangle,'e']
                        target_population = internal_population_dict[g_index_seed,'e']

                        curr_connection = Connection(source_population,target_population,nsyn = Cell_type_num['e'],nsyn_post = Cell_type_num['e'], 
                                             weights = ge2e * short_range_exc2exc[g_index_seed,i_up_triangle],probs = 1.0,conn_type = 'ShortRange',v_min = -1.0,v_max = 1.0,dv = dv)

                        DEE[g_index_seed,i_up_triangle] += ge2e * short_range_exc2exc[g_index_seed,i_up_triangle]

                        connection_list.append(curr_connection)
                    
                    # source_exc, target_inh
                    if(short_range_exc2inh[i_up_triangle,g_index_seed]>(4.7*1e-6)):
                        source_population = internal_population_dict[g_index_seed,'e']
                        target_population = internal_population_dict[i_up_triangle,'i']

                        curr_connection = Connection(source_population,target_population,nsyn = Cell_type_num['e'],nsyn_post = Cell_type_num['i'], 
                                             weights = ge2i * short_range_exc2inh[i_up_triangle,g_index_seed],probs = 1.0,conn_type = 'ShortRange',v_min = -1.0,v_max = 1.0,dv = dv)


                        DIE[i_up_triangle,g_index_seed] += ge2i * short_range_exc2inh[i_up_triangle,g_index_seed]

                        connection_list.append(curr_connection)
                    
                    if(short_range_exc2inh[g_index_seed,i_up_triangle]>(4.7*1e-6)):
                        source_population = internal_population_dict[i_up_triangle,'e']
                        target_population = internal_population_dict[g_index_seed,'i']

                        curr_connection = Connection(source_population,target_population,nsyn = Cell_type_num['e'],nsyn_post = Cell_type_num['i'], 
                                             weights = ge2i * short_range_exc2inh[g_index_seed,i_up_triangle],probs = 1.0,conn_type = 'ShortRange',v_min = -1.0,v_max = 1.0,dv = dv)

                        DIE[g_index_seed,i_up_triangle] += ge2i * short_range_exc2inh[g_index_seed,i_up_triangle]

                        connection_list.append(curr_connection)
                    
                    # source_inh, target_exc
                    if(short_range_inh2exc[i_up_triangle,g_index_seed]>(4.7*1e-6)):
                        source_population = internal_population_dict[g_index_seed,'i']
                        target_population = internal_population_dict[i_up_triangle,'e']

                        curr_connection = Connection(source_population,target_population,nsyn = Cell_type_num['i'],nsyn_post = Cell_type_num['e'], 
                                             weights = gi2e * short_range_inh2exc[i_up_triangle,g_index_seed],probs = 1.0,conn_type = 'ShortRange',v_min = -1.0,v_max = 1.0,dv = dv)

                        DEI[i_up_triangle,g_index_seed] += gi2e * short_range_inh2exc[i_up_triangle,g_index_seed]

                        connection_list.append(curr_connection)
                    if(short_range_inh2exc[g_index_seed,i_up_triangle]>(4.7*1e-6)):
                        source_population = internal_population_dict[i_up_triangle,'i']
                        target_population = internal_population_dict[g_index_seed,'e']

                        curr_connection = Connection(source_population,target_population,nsyn = Cell_type_num['i'],nsyn_post = Cell_type_num['e'], 
                                             weights = gi2e * short_range_inh2exc[g_index_seed,i_up_triangle],probs = 1.0,conn_type = 'ShortRange',v_min = -1.0,v_max = 1.0,dv = dv)

                        DEI[g_index_seed,i_up_triangle] += gi2e * short_range_inh2exc[g_index_seed,i_up_triangle]

                        connection_list.append(curr_connection)
                    
                    # source_inh, target_inh
                    if(short_range_inh2inh[i_up_triangle,g_index_seed]>(4.7*1e-6)):
                        source_population = internal_population_dict[g_index_seed,'i']
                        target_population = internal_population_dict[i_up_triangle,'i']

                        curr_connection = Connection(source_population,target_population,nsyn = Cell_type_num['i'],nsyn_post = Cell_type_num['i'], 
                                             weights = gi2i * short_range_inh2inh[i_up_triangle,g_index_seed],probs = 1.0,conn_type = 'ShortRange',v_min = -1.0,v_max = 1.0,dv = dv)

                        DII[i_up_triangle,g_index_seed] += gi2i * short_range_inh2inh[i_up_triangle,g_index_seed]

                        connection_list.append(curr_connection)
                    if(short_range_inh2inh[g_index_seed,i_up_triangle]>(4.7*1e-6)):
                        source_population = internal_population_dict[i_up_triangle,'i']
                        target_population = internal_population_dict[g_index_seed,'i']

                        curr_connection = Connection(source_population,target_population,nsyn = Cell_type_num['i'],nsyn_post = Cell_type_num['i'], 
                                             weights = gi2i * short_range_inh2inh[g_index_seed,i_up_triangle],probs = 1.0,conn_type = 'ShortRange',v_min = -1.0,v_max = 1.0,dv = dv)

                        DII[g_index_seed,i_up_triangle] += gi2i * short_range_inh2inh[g_index_seed,i_up_triangle]


                        connection_list.append(curr_connection)
    """
    """        
    return connection_list,DEE,DIE,DEI,DII


def create_longrange_connection(Net_settings,sigma_hyplr):
	lrtarget_xaxis, lrtarget_yaxis = 0.0,0.0
    # center 0,0
	xhyp_tt = Net_settings['xhyp'] * 1
	lenx    = xhyp_tt * 2 + 1
	yhyp_tt = Net_settings['yhyp'] * 1 # 0#
	leny    = yhyp_tt * 2 + 1 #1#
    # here we only care about x-axis, not y-axis perpendicular to exact x-axis 

	dxx_hyp2 = Net_settings['dxx_hyp']**2

	xhyp_range = np.reshape(np.arange(-xhyp_tt,xhyp_tt+1),(lenx,1))
	xhyp_range = np.repeat(xhyp_range,leny,axis = 1)

	yhyp_range = np.reshape(np.arange(-yhyp_tt,yhyp_tt+1),(1,leny))
	yhyp_range = np.repeat(yhyp_range,lenx,axis = 0)

	lrhyp_xdis = np.abs(xhyp_range - lrtarget_xaxis * np.ones_like(xhyp_range))
	lrhyp_xdis = lrhyp_xdis**2 * dxx_hyp2

	lrhyp_ydis = np.abs(yhyp_range - lrtarget_yaxis * np.ones_like(yhyp_range))
	lrhyp_ydis = lrhyp_ydis**2 * dxx_hyp2

	lrhyp_dist = np.sqrt(lrhyp_xdis + lrhyp_ydis) # squart

	# normalization long-range connections for ei and ee
	_normalization_factor = np.sqrt(2*np.pi)

	cen_amp = {}
	cen_amp['lr','exc2exc'] = np.zeros((xhyp_tt+1,yhyp_tt+1))
#	print('xhyp:',xhyp_tt,'yhyp:',yhyp_tt)
	cen_amp['lr','exc2inh'] = np.zeros((xhyp_tt+1,yhyp_tt+1))
	# long-range excitatory to excitatory
	mean,sigma = 0.0,sigma_hyplr['long_range_exc_exc']
	exp_amp    = np.exp(-np.power((lrhyp_dist - mean)/sigma, 2)/2.0) / (sigma * _normalization_factor)
	exp_amp[xhyp_tt,yhyp_tt] = 0.0
	sum_amp = np.squeeze(np.reshape(exp_amp,(lenx*leny,1)))
	sum_amp = np.sum(sum_amp)
	exp_amp /= sum_amp
	cen_amp['lr','exc2exc'] = exp_amp[xhyp_tt:,yhyp_tt:]

	# long-range excitatory to inhibitory
	mean,sigma = 0.0,sigma_hyplr['long_range_exc_inh']
	exp_amp    = np.exp(-np.power((lrhyp_dist - mean)/sigma, 2)/2.0) / (sigma * _normalization_factor)
	exp_amp[xhyp_tt,yhyp_tt] = 0.0
	sum_amp = np.squeeze(np.reshape(exp_amp,(lenx*leny,1)))
	sum_amp = np.sum(sum_amp)
	exp_amp /= sum_amp
	cen_amp['lr','exc2inh'] = exp_amp[xhyp_tt:,yhyp_tt:]

	return lrhyp_xdis,lrhyp_ydis,lrhyp_dist,cen_amp



Net_settings,Hypm,Pham,Orim,index_2_loc = create_functional_columns(Net_settings,Fun_map_settings,ori)
(dxx,dyy) = (500.0/Net_settings['xn'],500.0/Net_settings['yn'])
dxx = dxx**2
dyy = dyy**2
global_x,global_y,global_dist = preprocess_for_distance(index_2_loc,dxx,dyy,Net_settings,Fun_map_settings)
cortical_norm = cortical_to_cortical_connection_normalization(global_dist,sr_sigma,lr_sigma,ori,Hypm,Net_settings,Fun_map_settings)

connection_list,DEE,DIE,DEI,DII = cortical_to_cortical_connection(background_population_dict,internal_population_dict,g_norm,0.0,Orim,Pham,
                                             cortical_norm)

"""
"""
simulation = Simulation(population_list, connection_list,Net_settings,Cell_type_num,DEE,DIE,DEI,DII,verbose=True)
(mEbin_ra,mIbin_ra,xEbin_ra,xIbin_ra,VEavgbin_ra,VIavgbin_ra,VEstdbin_ra,VIstdbin_ra,rEbin_ra,rIbin_ra,P_MFEbin_ra,nEbin_ra,nIbin_ra) = simulation.update(t0=t0, dt=dt, tf=tf)


# recording data
import time
import scipy.io as scio
ISOTIMEFORMAT='%Y%m%d%H%M%S'
filename=str(time.strftime(ISOTIMEFORMAT)) + '_large.mat'
scio.savemat(filename,{'mEbin_ra':mEbin_ra,'mIbin_ra':mIbin_ra,'xEbin_ra':xEbin_ra,'xIbin_ra':xIbin_ra,'rEbin_ra':rEbin_ra,'rIbin_ra':rIbin_ra,'P_MFEbin_ra':P_MFEbin_ra,'nEbin_ra':nEbin_ra,'nIbin_ra':nIbin_ra}) 
filename=str(time.strftime(ISOTIMEFORMAT)) + 'V_large.mat'
scio.savemat(filename,{'VEavgbin_ra':VEavgbin_ra,'VIavgbin_ra':VIavgbin_ra,'VEstdbin_ra':VEstdbin_ra,'VIstdbin_ra':VIstdbin_ra}) 
fileparamname=str(time.strftime(ISOTIMEFORMAT)) + '_large_params.mat'
scio.savemat(fileparamname, {'DEE':DEE,'DEI':DEI,'DIE':DIE,'DII':DII,'active_brightened':active_brightened,'active_darken':active_darken})

