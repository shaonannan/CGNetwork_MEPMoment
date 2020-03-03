import numpy as np
import scipy.linalg as spla
import scipy.stats as sps
import scipy.integrate as spi
import bisect
from scipy.optimize import minimize
from scipy import special
import time
import scipy.misc as sm
import scipy.special as ss

''' for MFE algorithm
'''
def getsamples(Vedges,rho,Nsamples):
    rho = np.squeeze(rho)
    rho = [0] + rho
    rho = rho/np.sum(rho)
    L   = len(rho)
    F   = np.cumsum(rho)
    [Fv,Fi,Fj]  = np.unique(F,return_index = True,return_inverse = True)
    # print('F_indx:',np.shape(F[Fi]),'inverse_indx:',np.shape(Vedges[Fi]))
    output = np.interp(np.random.random(Nsamples),F[Fi],Vedges[Fi])
    
    return output

def rho_EQ(Vs,D,V):
    Rv = np.copy(V)
    (vT,vR) = (1.0,0.0)
    tmpg = np.greater(V,vR)
    indp = (np.where(tmpg))
    sqrtD  = np.sqrt(D)
    np.seterr(all='ignore')
    try:
        intovT  = special.dawsn((vT-Vs)/sqrtD)*np.exp((vT-Vs)**2/D)
        intovSD = special.dawsn(-Vs/sqrtD)*np.exp(Vs**2/D)
        Rv[indp[0][:]] = -special.dawsn((V[indp[0][:]]-Vs)/sqrtD)+np.exp(-(V[indp[0][:]]-Vs)**2/D)*intovT
        
        if(indp[0][0]>1):
            Rv[0:indp[0][0]] = np.exp(-np.square(V[0:indp[0][0]]-Vs)/D)*(-intovSD + intovT)
        tmpl = np.less(V,-2.0/3.0)
        indp = np.where(tmpl)
        Rv[indp[0][:]] = 0.0  
        sum_c = (V[2]-V[1])*np.sum(Rv)
        Rv = Rv/sum_c
    except:
        sum_c = (V[2]-V[1])*np.sum(Rv)
        Rv = Rv/sum_c        
    return (Rv,sum_c)

def optfun(lambda_u,mu,x,Pq,fin,gamma):
    lambda_u = lambda_u[:]
    k  = np.size(mu)
    tt = np.zeros(k+1)
    tt[0] = 1
    tt[1:k+1]  = mu[:]
    dx = x[1]-x[0]
    N  =np.size(lambda_u)

    p  = Pq*np.exp(np.dot(fin[:,0:N],lambda_u), dtype=np.float64)
    f  = dx*np.sum(p)-np.dot(np.reshape(tt,[1,k+1]),lambda_u)
    return f

def get_v_edges(v_min,v_max,dv):
    # Used for voltage-distribution and discretization
    edges = np.concatenate((np.arange(v_min,v_max,dv),[v_max]))
    edges[np.abs(edges) < np.finfo(np.float).eps] = 0
    return edges

def getMFE_ifdyn(verbose,VE,VI,DEE,DEI,DIE,DII,DEY,DIY,pop_idx_E,pop_idx_I):
    ''' assume that voltages are distinct
    '''
    (TAU,VT,VR) = (20.0,1.0,0.0)
    (NE,NPATCH)  = np.shape(np.squeeze(VE))
    (NI,NPATCH1) = np.shape(np.squeeze(VI))

    ''' checking '''
    if NPATCH!=NPATCH1:
        print('DIMENSION MISMATCH!')
    VE,pop_idx_E = np.reshape(VE.T,(NE*NPATCH,1)),np.reshape(pop_idx_E.T,(NE*NPATCH,1))
    VI,pop_idx_I = np.reshape(VI.T,(NI*NPATCH1,1)),np.reshape(pop_idx_I.T,(NI*NPATCH1,1))
    VE,VI = np.squeeze(VE),np.squeeze(VI)
    pop_idx_E,pop_idx_I = np.squeeze(pop_idx_E),np.squeeze(pop_idx_I)
    ''' if use squeeze, the could get 1-D I_fired and E_fired '''

    VEj = np.argsort(VE)
    VE  = np.sort(VE)
    pop_idx_E = pop_idx_E[VEj]
    VIj = np.argsort(VI)
    VI  = np.sort(VI)
    pop_idx_I = pop_idx_I[VIj]
    
    VEr = np.argsort(VEj)
    VEjs= np.sort(VEj)
    VIr = np.argsort(VIj)
    VIjs= np.sort(VIj)
    
    VE_orig,VI_orig = VE.copy(),VI.copy() 
    ''' have already sorted '''
    
    '''
    find E_fired/remaining I_fired/remaining
    '''
    I_fired = np.where(VI>=VT)
    I_fired = np.squeeze(I_fired)
    LI      = np.size(I_fired)
    I_remaining = np.where(VI<VT)
    I_remaining = np.squeeze(I_remaining)
    accumulated_DII,accumulated_DEI = np.zeros((NPATCH*NI,1)),np.zeros((NPATCH*NE,1))
    could_use_min_to_I,could_use_min_to_E = np.zeros((NPATCH*NI,1)),np.zeros((NPATCH*NE,1))
    for i in range(LI):
        ''' all about Inh '''
        if LI == 1:
            type_I = pop_idx_I[I_fired]
        else:
            type_I = pop_idx_I[I_fired[i]]
        target_I_head = pop_idx_I
        target_I_remaining = np.squeeze(pop_idx_I[I_remaining])
        ''' all about Exc '''
        target_E_head = pop_idx_E
        accumulated_DII += np.reshape(DII[target_I_head,type_I],(len(target_I_head),1))
        could_use_min_to_I = np.column_stack((could_use_min_to_I,DII[target_I_head,type_I]))
        VI[I_remaining] -= np.squeeze(DII[target_I_remaining,type_I])
        accumulated_DEI += np.reshape(DEI[target_E_head,type_I],(len(target_E_head),1))
        could_use_min_to_E = np.column_stack((could_use_min_to_E,DEI[target_E_head,type_I]))
        VE -= DEI[target_E_head,type_I]
        
    E_fired = np.where(VE>=VT)
    E_fired = np.squeeze(E_fired)
    LE      = np.size(E_fired)
    E_remaining = np.where(VE<VT)
    E_remaining = np.squeeze(E_remaining)
    total_V_to_add_to_E,total_V_to_add_to_I = np.zeros((NPATCH*NE,1)),np.zeros((NPATCH*NI,1))
    could_use_add_to_E,could_use_add_to_I = np.zeros((NPATCH*NE,1)),np.zeros((NPATCH*NI,1))
    for i in range(LE):
        ''' all about Exc '''
        if LE == 1:
            type_E = pop_idx_E[E_fired]
        else:
            type_E = pop_idx_E[E_fired[i]]
        target_E_head = pop_idx_E
        ''' all about Inh '''
        target_I_head = pop_idx_I

        total_V_to_add_to_E += np.reshape(DEE[target_E_head,type_E],(len(target_E_head),1))
        could_use_add_to_E = np.column_stack((could_use_add_to_E,DEE[target_E_head,type_E]))
        total_V_to_add_to_I += np.reshape(DIE[target_I_head,type_E],(len(target_I_head),1))
        could_use_add_to_I = np.column_stack((could_use_add_to_I,DIE[target_I_head,type_E]))
    ''' START EACH CONDITIONS
    '''
    iter = 0
    while (max(np.squeeze(total_V_to_add_to_E)) >0)|(max(np.squeeze(total_V_to_add_to_I)) >0):
        iter+=1
        possible_E_spikes = np.where(VE[E_remaining]>=(VT - np.squeeze(total_V_to_add_to_E[E_remaining])))
        possible_I_spikes = np.where(VI[I_remaining]>=(VT - np.squeeze(total_V_to_add_to_I[I_remaining])))
        ce = np.shape(possible_E_spikes)[1]
        ci = np.shape(possible_I_spikes)[1]
        if (ce>0):          
            max_E,ind_E = max(VE[E_remaining]),np.argmax(VE[E_remaining])
            ind_E = E_remaining[ind_E]
        if(ci>0):
            max_I,ind_I = max(VI[I_remaining]),np.argmax(VI[I_remaining])
            ind_I = I_remaining[ind_I]

        if((ce<1) & (ci<1)):
            V_to_add_to_E = total_V_to_add_to_E
            V_to_add_to_I = total_V_to_add_to_I
            VE[E_remaining] += V_to_add_to_E[E_remaining,0]
            VI[I_remaining] += V_to_add_to_I[I_remaining,0]
            total_V_to_add_to_E = np.zeros_like(total_V_to_add_to_E)
            total_V_to_add_to_I = np.zeros_like(total_V_to_add_to_I)
        elif((ce>0)&(ci<1)):
            V_to_add_to_E = VT - max_E
            V_to_add_to_I = np.minimum(total_V_to_add_to_I,VT-max_E)
            #print('155 vti:',V_to_add_to_I)
            E_fired = np.append(E_fired,ind_E)
            E_remaining = np.setdiff1d(E_remaining,ind_E)
            LE += 1

            type_E = pop_idx_E[ind_E]
            target_E_head = pop_idx_E
            target_E_remaining = pop_idx_E[E_remaining]
            target_I_head = pop_idx_I
            target_I_remaining = pop_idx_I[I_remaining]

            VE[E_remaining] += V_to_add_to_E*np.ones_like(VE[E_remaining])
            VI[I_remaining] += np.squeeze(V_to_add_to_I[I_remaining])

            total_V_to_add_to_E = total_V_to_add_to_E - V_to_add_to_E + np.reshape(DEE[target_E_head,type_E],(len(target_E_head),1))
            could_use_add_to_E  = np.column_stack((could_use_add_to_E,DEE[target_E_head,type_E]))
            total_V_to_add_to_I = total_V_to_add_to_I - V_to_add_to_I + np.reshape(DIE[target_I_head,type_E],(len(target_I_head),1))
            could_use_add_to_I  = np.column_stack((could_use_add_to_I,DIE[target_I_head,type_E]))
        elif((ce<1)&(ci>0)):
            V_to_add_to_I = VT - max_I
            V_to_add_to_E = np.minimum(total_V_to_add_to_E,VT-max_I)
            type_I = pop_idx_I[ind_I]
            target_E_head = pop_idx_E
            target_E_remaining = pop_idx_E[E_remaining]
            VE[E_remaining] = VE[E_remaining] - np.squeeze(DEI[target_E_remaining,type_I]) + np.squeeze(V_to_add_to_E[E_remaining])
            accumulated_DEI += np.reshape(DEI[target_E_head,type_I],(len(target_E_head),1))
            ''' I-fired '''
            I_fired = np.append(I_fired,ind_I)
            I_remaining = np.setdiff1d(I_remaining, ind_I)
            LI += 1
            target_I_head = pop_idx_I
            target_I_remaining = pop_idx_I[I_remaining]

            VI[I_remaining] = VI[I_remaining] - np.squeeze(DII[target_I_remaining,type_I]) + V_to_add_to_I * np.ones_like(VI[I_remaining])
            accumulated_DII += np.reshape(DII[target_I_head,type_I],(len(target_I_head),1))

            total_V_to_add_to_E -= V_to_add_to_E
            total_V_to_add_to_I -= V_to_add_to_I
            #print('198: ',total_V_to_add_to_I.T)
        elif((ce>0)&(ci>0)):
            ''' E-fired '''
            temp_add_E = 0.0
            v_theo_add_E = VT - VE_orig[ind_E] + accumulated_DEI[ind_E]
            temp_E_ratio = 0.0
            new_could_use_add_E = 0

            use_voltage_target_E = np.squeeze(could_use_add_to_E[ind_E,:])
            for idxE in range(len(use_voltage_target_E)):
                temp_add_E += use_voltage_target_E[idxE]
                if temp_add_E>= v_theo_add_E:
                    temp_E_residual = temp_add_E - v_theo_add_E
                    temp_E_cross    = use_voltage_target_E[idxE] - temp_E_residual
                    temp_E_ratio    = temp_E_cross/use_voltage_target_E[idxE]
                    temp_E_ratio_check = 1.0 - temp_E_residual/use_voltage_target_E[idxE]
#                    if temp_E_ratio!= temp_E_ratio_check:
#                        print('Excitatory ratio mismatch! a: ',temp_E_ratio,' b: ',temp_E_ratio_check)
                    new_could_use_add_E = idxE + temp_E_ratio
                    break

            ''' I-fired '''
            temp_add_I = 0.0
            v_theo_add_I = VT-VI_orig[ind_I] +accumulated_DII[ind_I]
            temp_I_ratio = 0.0
            new_could_use_add_I = 0

            use_voltage_target_I = np.squeeze(could_use_add_to_I[ind_I,:])
            for idxI in range(len(use_voltage_target_I)):
                temp_add_I += use_voltage_target_I[idxI]
                if temp_add_I >= v_theo_add_I:
                    temp_I_residual = temp_add_I - v_theo_add_I
                    temp_I_cross = use_voltage_target_I[idxI] - temp_I_residual
                    temp_I_ratio = temp_I_cross/use_voltage_target_I[idxI]
                    temp_I_ratio_check = 1.0 - temp_I_residual/use_voltage_target_I[idxI]
#                    if temp_I_ratio!=temp_I_ratio_check:
#                        print('Inhibitory ratio mismatch! a: ',temp_I_ratio,' b: ',temp_I_ratio_check)
                    new_could_use_add_I = idxI + temp_I_ratio
                    break
            if new_could_use_add_E < new_could_use_add_I:
                ''' E-fired '''
                E_fired = np.append(E_fired,ind_E)
                E_remaining = np.setdiff1d(E_remaining,ind_E)
                LE += 1
                V_to_add_to_E = VT - max_E
                V_to_add_to_I = np.minimum(total_V_to_add_to_I,VT-max_E)
                type_E = pop_idx_E[ind_E]
                target_E_head = pop_idx_E
                target_E_remaining = pop_idx_E[E_remaining]

                target_I_head = pop_idx_I
                target_I_remaining = pop_idx_I[I_remaining]

                could_use_add_to_E = np.column_stack((could_use_add_to_E,DEE[target_E_head,type_E]))
                total_V_to_add_to_E = total_V_to_add_to_E + np.reshape(DEE[target_E_head,type_E],(len(target_E_head),1)) - V_to_add_to_E
                could_use_add_to_I = np.column_stack((could_use_add_to_I,DIE[target_I_head,type_E]))
                total_V_to_add_to_I = total_V_to_add_to_I + np.reshape(DIE[target_I_head,type_E],(len(target_I_head),1)) - V_to_add_to_I

                VE[E_remaining] = VE[E_remaining] + V_to_add_to_E * np.ones_like(VE[E_remaining])
                VI[I_remaining] = VI[I_remaining] + np.squeeze(V_to_add_to_I[I_remaining])
            elif new_could_use_add_I <= new_could_use_add_E:
                ''' I-fired '''
                I_fired = np.append(I_fired,ind_I)
                I_remaining = np.setdiff1d(I_remaining,ind_I)
                LI += 1
                V_to_add_to_I = VT - max_I
                V_to_add_to_E = np.minimum(total_V_to_add_to_E,VT-max_I)

                type_I = pop_idx_I[ind_I]
                target_I_head = pop_idx_I
                target_I_remaining = pop_idx_I[I_remaining]

                target_E_head = pop_idx_E
                target_E_remaining = pop_idx_E[E_remaining]

                could_use_min_to_E = np.column_stack((could_use_min_to_E,DEI[target_E_head,type_I]))
                total_V_to_add_to_E = total_V_to_add_to_E - V_to_add_to_E
                accumulated_DEI += np.reshape(DEI[target_E_head,type_I],(len(target_E_head),1))

                could_use_min_to_I = np.column_stack((could_use_min_to_I,DII[target_I_head,type_I]))
                total_V_to_add_to_I = total_V_to_add_to_I - V_to_add_to_I
                accumulated_DII += np.reshape(DII[target_I_head,type_I],(len(target_I_head),1))

                VE[E_remaining] = VE[E_remaining] - np.squeeze(DEI[target_E_remaining,type_I]) + np.squeeze(V_to_add_to_E[E_remaining])
                VI[I_remaining] = VI[I_remaining] - np.squeeze(DII[target_I_remaining,type_I]) + V_to_add_to_I*np.ones_like(VI[I_remaining])

    VE,VI = np.reshape(VE,(NE*NPATCH,1)),np.reshape(VI,(NI*NPATCH,1))
    E_fired,I_fired = np.reshape(E_fired,(np.size(E_fired),1)),np.reshape(I_fired,(np.size(I_fired),1))
    ncurrE,ncurrI = len(VE),len(VI)
    VE_pre,VI_pre = VE,VI 
    VE_pre[ncurrE-LE:,0] = 1.0
    VI_pre[ncurrI-LI:,0] = 1.0

    VE_pos,VI_pos = VE,VI 
    VE_pos[ncurrE-LE:,0] = 0.0
    VI_pos[ncurrI-LI:,0] = 0.0

    E_fired,I_fired = VEj[E_fired], VIj[I_fired]
    VE_pre,VI_pre = VE_pre[VEr],VI_pre[VIr]
    VE_pos,VI_pos = VE_pos[VEr],VI_pos[VIr]

    pop_idx_E_col,pop_idx_I_col = pop_idx_E[VEr],pop_idx_I[VIr]

    VE_pre = np.reshape(VE_pre,(NPATCH,NE))
    VE_pre = VE_pre.T
    VI_pre = np.reshape(VI_pre,(NPATCH,NI))
    VI_pre = VI_pre.T
    VE_pos = np.reshape(VE_pos,(NPATCH,NE))
    VE_pos = VE_pos.T
    VI_pos = np.reshape(VI_pos,(NPATCH,NI))
    VI_pos = VI_pos.T

    pop_idx_E_col = np.reshape(pop_idx_E_col,(NPATCH,NE))
    pop_idx_E_col = pop_idx_E_col.T
    pop_idx_I_col = np.reshape(pop_idx_I_col,(NPATCH,NI))
    pop_idx_I_col = pop_idx_I_col.T

    return (E_fired,I_fired,VE_pre,VI_pre,VE_pos,VI_pos)


def psample(ldt):
    if ldt>5:
        outspike = np.maximum(0,round(ldt + np.sqrt(ldt)*np.random.randn()))
    else:
        kra = np.arange(15)
        pra = np.cumsum((ldt**kra)*np.exp(-ldt)/ss.factorial(kra))
        pra = pra[::-1]
        pra = np.append(pra,-1)
        pra = pra[::-1]
        pra = np.append(pra,2)
        mininner = np.minimum(np.random.random(),pra[-1])
        idx = np.where(pra-mininner<0)
        maxidx = max(idx[0][:])
        outspike = maxidx
    return outspike