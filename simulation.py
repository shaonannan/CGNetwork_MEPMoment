from connectiondistributioncollection import ConnectionDistributionCollection
import time
import numpy as np
import utilities as util
import matplotlib.pyplot as plt
import time
import scipy.io as scio 


class Simulation(object):
    """
    Parameters:
    list :
        All sub-population (cluster)
        All connection (cluster)
        [type of both is 'List', which is changable variable, and could be changed]
        
    generate after initiate(by hand)
        connection_distribution
        connection_distribution_list
        [the differences between connection, connection_distribution and connection_distribution_list are
        connection ---> the component of 'connection_list', record all information and related information and object,like source and others
        connection_distribution --> this variable is a preparation variable for further processing, each 'connection' could generate a 
        class 'connecton_distribution' and then, using weight,syn,prob, could calculate flux_matrix and threshold
        each 'connection_distribution' item is defined by 'weight''syn ''prob', items with identical symbol will be classified to the same
        distribution
        connection_distribution_list --> this is a 'basket', store all unique connections(definition of unique: unique symbol
        'weight','syn','prob' no matter the target/source population)
    """
    def __init__(self,population_list,connection_list,Net_settings,Cell_type_num,DEE,DIE,DEI,DII,verbose=True):
        
        self.verbose = verbose
        self.population_list = population_list
        self.connection_list = [c for c in connection_list if c.nsyn!=0.0]
        self.Net_settings    = Net_settings
        tfinal = Net_settings['Final_time']
        dt     = Net_settings['dt']
        self.ntt = int(tfinal/dt)
        self.m_record = None
        ''' all for MFE '''
        self.VE,self.VI = None,None
        self.Vedges,self.Vbins = None,None
        self.NE,self.NI = Cell_type_num['e'],Cell_type_num['i']
        self.MFE_num  = 0
        self.MFE_flag = 0
        
        self.DEE = DEE
        self.DEI = DEI
        self.DIE = DIE
        self.DII = DII

    
    def initialize(self,t0=0.0):
        """
        initialize by hand, first put all sub-population and connection-pair
        """
        ''' initialize P_MFE '''
        ''' at first, we only use NHYP as NPATCH '''
        self.iteration_max = self.ntt+100
        iteration_max = self.iteration_max
        self.tbin_tmp = 0 # initial
        self.tbinsize = 1.0
        dtperbin = int(self.tbinsize/self.dt)
        self.dtperbin = dtperbin
        iteration_bin = int(iteration_max/dtperbin)
        NPATCH = self.Net_settings['nmax']
        NE,NI  = self.NE,self.NI
        self.VE,self.VI = np.zeros((NE,NPATCH)),np.zeros((NI,NPATCH))
        # DTBIN_RECORD_FLAG
        self.tbin_ra = np.zeros((iteration_max,1))

        self.mEbin_ra = np.zeros((iteration_bin,NPATCH))
        self.mIbin_ra = np.zeros_like(self.mEbin_ra)
        self.xEbin_ra = np.zeros_like(self.mEbin_ra)
        self.xIbin_ra = np.zeros_like(self.xEbin_ra)
        
        self.nEbin_ra = np.zeros_like(self.xEbin_ra)
        self.nIbin_ra = np.zeros_like(self.xEbin_ra)
        
        ''' also recording possible MFE as well as effective MFE'''
        # possible MFE-prob and index
        self.P_MFEbin_ra = np.zeros_like(self.xIbin_ra)
        self.P_MFE_ra    = np.zeros((iteration_max,1))
        self.idx_MFE_ra  = np.zeros((iteration_max,1))
        # effective MFE-prob and index
        self.P_MFE_eff   = np.zeros((iteration_max,1))
        self.idx_MFE_eff = np.zeros((iteration_max,1))
        
        self.rEbin_ra = np.zeros((NPATCH,2000,iteration_bin))
        self.rIbin_ra = np.zeros_like(self.rEbin_ra)

        self.VEavgbin_ra = np.zeros_like(self.P_MFEbin_ra)
        self.VIavgbin_ra = np.zeros_like(self.VEavgbin_ra)
        self.VEstdbin_ra = np.zeros_like(self.VIavgbin_ra)
        self.VIstdbin_ra = np.zeros_like(self.VEstdbin_ra)

        self.rE,self.rI  = None,None
        self.NPATCH = NPATCH
        
        self.LE_ra = np.zeros((iteration_max,NPATCH))
        self.LI_ra = np.zeros_like(self.LE_ra)
        
        DEE,DIE,DEI,DII = self.DEE,self.DIE,self.DEI,self.DII

        vT = 1.0
        dv = self.Net_settings['dv']
        self.Vedges = util.get_v_edges(-1.0,1.0,dv)
        ''' bins = edges - 1'''
        # in internal , rhov length len(self.Vbins), len(Vedges)-1
        self.Vbins = 0.5*(self.Vedges[0:-1] + self.Vedges[1:]) 
        Vedges = self.Vedges 
        Vbins  = self.Vbins
        idx_vT = len(Vedges)-1 #len(Vbins)
        idx_kickE,idx_kickI = np.zeros((NPATCH,NPATCH),dtype=int),np.zeros((NPATCH,NPATCH),dtype=int)
        for it in range(self.NPATCH):
            for js in range(self.NPATCH):
                value_kickE = vT - DEE[it,js]
                value_kickI = vT - DIE[it,js]
                Ind_k1  = np.where(Vedges>value_kickE)
                IndI_k1 = np.where(Vedges>value_kickI) 
                if np.shape(Ind_k1)[1]>0:
                    idx_kickE[it,js]  = Ind_k1[0][0]
                else:
                    idx_kickE[it,js]  = idx_vT
                if np.shape(IndI_k1)[1]>0:
                    idx_kickI[it,js]  = IndI_k1[0][0]
                else:
                    idx_kickI[it,js]  = idx_vT
        
        self.idx_kickE,self.idx_kickI = idx_kickE,idx_kickI
        self.idx_vT   = idx_vT
        self.MFE_pevent = np.zeros(self.NPATCH)
        self.p_single = np.zeros(self.NPATCH)
        self.rE = np.zeros((len(self.Vbins),self.NPATCH))
        self.rI = np.zeros_like(self.rE)

        # An connection_distribution_list (store unique connection(defined by weight,syn,prob))
        self.connection_distribution_collection = ConnectionDistributionCollection() # this is 
        self.t = t0

        # Matrix to record 
        numCGPatch = self.Net_settings['nmax'] * 2 # excitatory and inhibitory
        # 2 * numCGPatch = External Population and Recurrent Population
        # set Matrix to record only Internal Population
        self.m_record = np.zeros((numCGPatch+1, self.ntt + 10))
        self.v_record = np.zeros_like(self.m_record)
        
        # put all subpopulation and all connections into the same platform
        for subpop in self.population_list:
            subpop.simulation = self    # .simulation = self(self is what we called 'simulation')
        for connpair in self.connection_list:
            connpair.simulation = self
            
        # initialize population_list, calculate         
        for p in self.population_list:
            p.initialize()      # 2   
        
        for c in self.connection_list:
            c.initialize()      # 1
  
    def update(self,t0,dt,tf):
        self.dt = dt
        self.tf = tf   
        # initialize:
        start_time = time.time()
        self.initialize(t0)
        self.initialize_time_period = time.time()-start_time
        
        # start_running
        start_time = time.time()
        counter = 0
        numCGPatch = self.Net_settings['nmax']*2
        '''
        First 
        '''
        Vbins,Vedges,NPATCH = self.Vbins,self.Vedges,self.NPATCH
        while self.t < self.tf:
            # refresh current time as well as current time-step
            self.t+=self.dt
            self.tbin_tmp = int(np.floor(self.t/self.tbinsize))

            #if self.verbose: print ('time: %s' % self.t)
            ind_rec,idxE,idxI = 0,0,0   # start to accumulate index of hypercolumn
            for p in self.population_list:
                ''' updating OP 2 modes: updating under Moment/updating under MFE '''
                # updating under Moment- full
                p.USUALorMFE = 1
                ind_rec += 1
                '''
                Recording at first, before p.update(),
                rE and rI purely from(after) MFE should be recorded in rE/I(bin)_ra, rather
                than RvE from Moment
                '''
                # before Moment iteration
                if(ind_rec>numCGPatch): # means internal-population, not external-population
                    if p.ei_pop == 'e':
                        ''' Voltage distribution should be recorded each dt as well as each dtbin'''
                        # dtbin-recording
                        self.VEavgbin_ra[self.tbin_tmp,idxE] += p.v1*dt
                        self.VEstdbin_ra[self.tbin_tmp,idxE] += np.sqrt(p.v2-p.v1**2)*dt
                        self.rEbin_ra[idxE,:,self.tbin_tmp] += p.curr_rhov * self.dt
                        idxE +=1
                    else:
                        ''' Voltage distribution should be recorded each dt as well as each dtbin'''
                        # dtbin-recording
                        self.VIavgbin_ra[self.tbin_tmp,idxI] += p.v1*dt
                        self.VIstdbin_ra[self.tbin_tmp,idxI] += np.sqrt(p.v2-p.v1**2)*dt
                        self.rIbin_ra[idxI,:,self.tbin_tmp] += p.curr_rhov * self.dt
                        idxI +=1
                p.update()
                '''
                when using USUALorMFE==1
                updating rhov as well as firing rate;
                
                next, recording firing rate mE/I in mE/I(bin)_ra
                [but not rE/I(bin)_ra]
                
                and also, RvE/I were extracted out from p-list, which were used
                to calculate MFE probability                   
                
                '''

                if(counter>0):
                    if(ind_rec>numCGPatch):
                        if p.ei_pop == 'e': 
                            continue
            ind_rec,idxE,idxI = 0,0,0
            for p in self.population_list:
                ind_rec += 1
                if(counter>0):
                    if(ind_rec>numCGPatch):
                        if p.ei_pop == 'e': 
                            '''
                            and also extract curr_rhov to calculate P_MFE
                            '''
                            self.rE[:,idxE] = p.curr_rhov
                            '''
                            here, recording new firing rate 
                            mE/I_ra
                            and also extracting curr_rhov to calculate PMFE
                            '''
                            self.p_single[idxE] = p.curr_firing_rate * self.dt * p.NumCell
                            idxE += 1
                        else:
                            self.rI[:,idxI] = p.curr_rhov
                            idxI += 1
                
            
            ''' calculating P_MFE for each time step '''
            ''' vedges is new for calculate MFE '''
            NE,NI = self.NE,self.NI
            NPATCH = self.NPATCH
            h = self.Vedges[1]- self.Vedges[0]
            local_pevent = 1.0
            
            for isource in range(self.NPATCH):
                local_pevent = 1.0
                for jt in range(self.NPATCH):
                    kickE,kickI = self.idx_kickE[jt,isource],self.idx_kickI[jt,isource]
                    idx_vT   = self.idx_vT
                    ''' Excitatory '''
                    trhov    = np.squeeze(self.rE[:,jt])
                    if isource!=jt:
                        Nup = NE 
                    else:
                        Nup = NE
                    
                    prob_event = (1.0 - np.sum(np.squeeze(trhov[kickE:idx_vT])) * h) ** Nup 
                    local_pevent *= prob_event
                    ''' Inhibitory '''
                    trhov    = np.squeeze(self.rI[:,jt])
                    Nup = NI
                    prob_event = (1.0 - np.sum(np.squeeze(trhov[kickI:idx_vT])) * h) ** Nup 
                    local_pevent *= prob_event
                self.MFE_pevent[isource] = self.p_single[isource] * (1-local_pevent)  

            ''' then prepare for MFE '''
            MFE_pevent_max = max(self.MFE_pevent)   # choose the maximum-value from all possible
            # trigger-pattern
            ''' recording MFE-probability '''
            # dt-recording
            self.P_MFE_ra[counter] = MFE_pevent_max
            # dt-bin recording probability of MFE 
            self.P_MFEbin_ra[self.tbin_tmp,0] +=  MFE_pevent_max * dt
            idx_pevent_max = np.argmax(self.MFE_pevent)
            ''' could also record which excitatory population appropriately trigger the MFE '''
            self.idx_MFE_ra[counter] = idx_pevent_max
            self.MFE_flag = 0
            local_pevent_d = np.random.random()
            if local_pevent_d < 0:#MFE_pevent_max:
                self.MFE_flag = 1
                self.MFE_num += 1
                self.idx_MFE_eff[self.MFE_num] = idx_pevent_max
                self.P_MFE_eff[self.MFE_num]   = MFE_pevent_max
                VE = np.zeros((self.NE,self.NPATCH))
                VI = np.zeros((self.NI,self.NPATCH))
                ''' distribution sampling '''
                for i in range(self.NPATCH):
                    V_sample = util.getsamples(np.squeeze(self.Vedges[1:]),np.squeeze(self.rE[:,i]),self.NE)
                    VE[:,i]  = np.squeeze(V_sample)
                    V_sample = util.getsamples(np.squeeze(self.Vedges[1:]),np.squeeze(self.rI[:,i]),self.NI)
                    VI[:,i]  = np.squeeze(V_sample)
                ''' trigger neuron '''
                VE[0:2,idx_pevent_max] = 1.0
                DEEu = self.DEE.copy()
                DIEu = self.DIE.copy()
                DEIu = self.DEI.copy()
                DIIu = self.DII.copy()
                
                pop_idx_E = np.zeros((self.NE,self.NPATCH),dtype=int)
                pop_idx_I = np.zeros((self.NI,self.NPATCH),dtype=int)
                for i in range(self.NPATCH):
                    pop_idx_E[:,i] = i
                    pop_idx_I[:,i] = i
                    
                (E_fired,I_fired,VE_pre,VI_pre,VE_pos,VI_pos) = util.getMFE_ifdyn(0,VE,VI,DEEu,DEIu,DIEu,DIIu,0,0,pop_idx_E,pop_idx_I)
                E_fired_num,I_fired_num = np.zeros((NE*NPATCH,1)),np.zeros((NI*NPATCH,1))
                E_fired_num[E_fired] = 1
                I_fired_num[I_fired] = 1
                E_fired_num = np.reshape(E_fired_num,(NE,NPATCH))
                I_fired_num = np.reshape(I_fired_num,(NI,NPATCH))
                E_ind_num = np.sum(E_fired_num,axis = 0)
                I_ind_num = np.sum(I_fired_num,axis = 0) 
                for i in range(NPATCH):
                    self.LE_ra[self.MFE_num,i] = E_ind_num[i]
                    self.LI_ra[self.MFE_num,i] = I_ind_num[i]
                VEpos,VIpos,VEpre,VIpre = VE_pos,VI_pos,VE_pre,VI_pre
                Vedges = self.Vedges
                Vbins  = self.Vbins
                h = Vbins[2]-Vbins[1]
                rE,rI = np.zeros((len(Vbins),NPATCH)),np.zeros((len(Vbins),NPATCH))
                '''
                resample and produce new voltage distribution after MFE correction
                '''
                for i in range(NPATCH):
                    VEposu = np.squeeze(VEpos[:,i])
                    rE_tmp,Vedge = np.histogram(VEposu, Vedges)
                    # new rhov
                    rE[:,i] = rE_tmp/(self.NE * h)#(np.sum(rE_tmp)*h)
                    
                    VIposu = np.squeeze(VIpos[:,i])
                    rI_tmp,Vedge = np.histogram(VIposu, Vedges)
                    # new rhov
                    rI[:,i] = rI_tmp/(self.NI *h) #(np.sum(rI_tmp)*h)
                '''
                refresh rhov for each population
                '''
                idxE,idxI = 0,0
                ind_rec   = 0
                for p in self.population_list:
                    ind_rec +=1
                    if(counter>0):
                        if(ind_rec>numCGPatch):
                            if p.ei_pop == 'e':
                                ''' rhovE[i]=0'''
                                p.rhov[:] = rE[:,idxE]
                                idxE += 1
                            else:
                                ''' rhovI[i]=0'''
                                p.rhov[:] = rI[:,idxI]
                                idxI += 1
                V1 = Vbins
                V2 = V1*Vbins
                V3 = V2*Vbins
                V4 = V3*Vbins   
                h = Vbins[2] - Vbins[1]
                vbarE = np.zeros(NPATCH)
                wbarE,vbar3E,vbar4E = np.zeros_like(vbarE),np.zeros_like(vbarE),np.zeros_like(vbarE)
                vbarI = np.zeros(NPATCH)
                wbarI,vbar3I,vbar4I = np.zeros_like(vbarI),np.zeros_like(vbarI),np.zeros_like(vbarI)
                for i in range(NPATCH):
                    rE_tmp = np.squeeze(rE[:,i])
                    rI_tmp = np.squeeze(rI[:,i])
                    vbarE[i] = np.sum(V1*rE_tmp ) * h
                    wbarE[i] = np.sum(V2*rE_tmp ) * h
                    vbar3E[i] = np.sum(V3*rE_tmp ) * h
                    vbar4E[i] = np.sum(V4*rE_tmp ) * h
                    
                    vbarI[i] = np.sum(V1*rI_tmp ) * h
                    wbarI[i] = np.sum(V2*rI_tmp ) * h
                    vbar3I[i] = np.sum(V3*rI_tmp ) * h
                    vbar4I[i] = np.sum(V4*rI_tmp ) * h
                ''' refresh data in simulation '''
                idxE,idxI = 0,0
                ind_rec   = 0
                '''
                set v1,v2,v3,v4 from outter
                USUALorMFE == 0,only change
                    VE/Is,DE/I (total.....), notice that before update, mE/I = 0,
                    p.firing_rate = 0
                    also,La0/1 equal to zero
                    
                    recalculate VE/Is DE/I
                    rhov_EQ(cause VE/Is and DE/I change)
                    
                    do not change rhov E/I
                    rhov have resampled after MFE, but didn't give it to p.rhov
                    so, give it to p-list, and do not change anymore (USUALorMFE==0)
                    
                    firing rate = 0 (USUALorMFE)                              
                '''
                for p in self.population_list:
                    ind_rec +=1
                    if(counter>0):
                        if(ind_rec>numCGPatch):
                            if p.ei_pop == 'e':
                                ''' 
                                print previous informatio, before MFE
                                '''
                                ''' me[i]=0'''
                                p.firing_rate = 0.0
                                ''' v1,v2,v3,v4 '''
                                p.v1,p.v2,p.v3,p.v4 = vbarE[idxE],wbarE[idxE],vbar3E[idxE],vbar4E[idxE]
                                ''' la0/1 '''
                                md_La = np.transpose([1,p.v1,p.v2])
                                p.La0,p.La1 = np.zeros_like(md_La),np.zeros_like(md_La)
                                idxE += 1
                            else:
                                p.firing_rate = 0.0
                                p.v1,p.v2,p.v3,p.v4 = vbarI[idxI],wbarI[idxI],vbar3I[idxI],vbar4I[idxI]
                                md_La = np.transpose([1,p.v1,p.v2])
                                p.La0,p.La1 = np.zeros_like(md_La),np.zeros_like(md_La)
                                idxI += 1
                    
                ''' after reset, calculate again'''
                ind_rec   = 0
                for p in self.population_list:
                    ind_rec +=1
                    if(counter>0):
                        if(ind_rec>numCGPatch):
                            p.USUALorMFE = 0
                            p.update()

            for c in self.connection_list:
                c.update()
            counter +=1
            
            ''' recording ! '''
                    # DTBIN_RECORD_FLAG
            self.tbin_ra[counter] = np.floor(self.t/self.tbinsize)
            tbin = int(np.floor(self.t/self.tbinsize))
            ind_rec,idxE,idxI   = 0,0,0
            for p in self.population_list:
                ind_rec +=1
                if(counter>0):
                    if(ind_rec>numCGPatch):
                        if p.ei_pop == 'e':
                            self.mEbin_ra[tbin,idxE] += (1-self.MFE_flag) * p.curr_firing_rate * NE * dt + self.MFE_flag * self.LE_ra[self.MFE_num,idxE]
                            self.xEbin_ra[tbin,idxE] += (1-self.MFE_flag) * util.psample(p.curr_firing_rate * NE * dt) + self.MFE_flag * self.LE_ra[self.MFE_num,idxE]
                            
                            self.nEbin_ra[tbin,idxE] += p.total_Inmda * NE * dt
                            idxE += 1
                        else:
                            self.mIbin_ra[tbin,idxI] += (1-self.MFE_flag) * p.curr_firing_rate * NE * dt + self.MFE_flag * self.LI_ra[self.MFE_num,idxI]
                            self.xIbin_ra[tbin,idxI] += (1-self.MFE_flag) * util.psample(p.curr_firing_rate * NE * dt) + self.MFE_flag * self.LI_ra[self.MFE_num,idxI]
                            
                            self.nIbin_ra[tbin,idxI] += p.total_Inmda * NI * dt
                            idxI += 1
                            
            ''' 
            # plotting the response curves
            if np.mod(counter,50) < 1:
                if np.mod(counter,50) == 0:
                    print("t_sum: ",counter * self.dt)
                for i in range(6,NPATCH,12):
                    print('Excitatory pop %d :%.4f'%(i,self.mEbin_ra[tbin,i]))
                    print('Inhibitory pop %d :%.4f'%(i,self.mIbin_ra[tbin,i]))
                    ttt = np.arange(tbin) * 1.0
                    plt.figure(220)
                    if (i<18):
                        plt.plot(ttt,self.VEavgbin_ra[:tbin,i],'r')
                    if (i>=18)&(i<32):
                        plt.plot(ttt,self.VEavgbin_ra[:tbin,i],'m')
                    if (i>=32):
                        plt.plot(ttt,self.VEavgbin_ra[:tbin,i],'y')
                    plt.xlim([0,int(self.tf)])
                    plt.ylim([0,1.2])
                    plt.pause(0.1)
                    plt.figure(222)
                    if (i<18):
                        plt.plot(ttt,self.VIavgbin_ra[:tbin,i],'r')
                    if (i>=18)&(i<32):
                        plt.plot(ttt,self.VIavgbin_ra[:tbin,i],'m')
                    if (i>=32):
                        plt.plot(ttt,self.VIavgbin_ra[:tbin,i],'y')
                    plt.xlim([0,int(self.tf)])
                    plt.ylim([0,1.2])
                    plt.pause(0.1)            
            '''

            ''' recording sub-time '''
            if np.mod(counter,500)==0:
                print("t_sum: ",counter * self.dt)
                icounter = np.ceil(counter/500)
                ip = int((icounter-1)*50)
                ic = int((icounter)*50)
                
                ISOTIMEFORMAT='%Y%m%d%H'
                filename=str(time.strftime(ISOTIMEFORMAT)) + str(icounter) + '.mat'
                scio.savemat(filename,{'mEbin_ra':self.mEbin_ra[ip:ic],'mIbin_ra':self.mIbin_ra[ip:ic],'xEbin_ra':self.xEbin_ra[ip:ic],'xIbin_ra':self.xIbin_ra[ip:ic],'VEavgbin_ra':self.VEavgbin_ra[ip:ic],'VIavgbin_ra':self.VIavgbin_ra[ip:ic],'VEstdbin_ra':self.VEstdbin_ra[ip:ic],'VIstdbin_ra':self.VIstdbin_ra[ip:ic],'nEbin_ra':self.nEbin_ra[ip:ic],'nIbin_ra':self.nIbin_ra[ip:ic]}) 
                    

        return self.mEbin_ra,self.mIbin_ra,self.xEbin_ra,self.xIbin_ra,self.VEavgbin_ra,self.VIavgbin_ra,self.VEstdbin_ra,self.VIstdbin_ra,self.rEbin_ra,self.rIbin_ra,self.P_MFEbin_ra,self.nEbin_ra,self.nIbin_ra