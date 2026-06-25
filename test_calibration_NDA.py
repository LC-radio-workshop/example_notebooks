import numpy as np
import matplotlib.pyplot as plt
from reading_nancay import read_newroutine
import glob

def calib_NDA_V1(filepath, thres_calib = 8E9, pplot=False, check=False):
    
    
    # Read data based on mode
    lh, rh, nda_time, freq = read_newroutine(filepath, pplot=False)
    resol = (nda_time[100]-nda_time[0])*3600/100 #0.5  # Resolution in seconds
    pos_freq=np.where((freq >=45) & (freq <=70))[0] # Avoid RFI
    
    print(f"Observing time interval: {nda_time.min()} - {nda_time.max()}")
    print(f"Resolution: {(nda_time[100]-nda_time[0])*3600/100}")
    # Determine calibration times
    #I . Calibration Phase
    #I.1 - Determine the Calibration Times

    # Calibration sequences consist in four steps of 10 seconds each: a complete calibration sequence lasts 40 seconds. 
    # The first step concerns very low intensities, which is usually difficult to distinguish from the background. 

    calib_duration = 9.2  # Duration of calibration in seconds
    skip_LastMinutes = 5*int(60/resol)
    axe_time = nda_time[:-skip_LastMinutes] # skip the last minutes, containing a longer calibration sequence
    Npt_PerStep = int(calib_duration / resol)
    
    network = 'LH'

    # Computes the time profile (averaged over the frequency range selected above)
    
    if network == 'RH':
        mean_int = np.mean(rh[:-skip_LastMinutes, pos_freq], axis=1)  
    else:
        mean_int = np.mean(lh[:-skip_LastMinutes, pos_freq], axis=1)  
    
    # Plot intermediate results if check is True
    
    # Look for signal above the threshold (sets in the variable thres_calib). 
    # UU: indices of the last calibration step (all points)
    uu = np.where(mean_int >= thres_calib )[0] 
    
    if len(uu) > 0:
        tmp = uu-np.roll(uu,1)
        aa = uu[np.where(tmp >= 1)[0]]
        aa=np.append(aa,uu[0])
        aa = np.array(sorted(aa))
        pos_skip = np.where((aa-3*Npt_PerStep) <0)[0]
        
        if len(pos_skip >0):
            aa = np.delete(aa, pos_skip)
            print('skip the first calibration sequence, which is not complete')
        
        # ----
        # Large solar activity may hide the calibration step. Based on the periodicity
        # of the calibration, retain only those multiple of the first calibration
        
        integers = np.arange(0, 7)
        values = axe_time[aa]-axe_time[aa[0]]
        margin = 0.003

        for integer in integers:
            candidates = np.where((values >= integer) & (values < integer + margin) | ((values <= integer) & (values > integer - margin)))[0]
            if len(candidates) >0:
                if integer == 0:
                    retained_index = candidates[0]
                else:
                    retained_index=np.append(retained_index,candidates[0])
        
        # -----------
        index_calib = aa[retained_index]
       
        if check:
            plt.figure(figsize=(12,5))
            plt.plot(axe_time,mean_int,marker='.')
            plt.xlabel('Time (Hours)')
            plt.axhline(y=thres_calib, color='r', linestyle='--', label='Y = 23')
            for idx in index_calib:
                plt.axvline(x=axe_time[idx], color='b', linestyle='-.', alpha=0.5)
            plt.ylabel('Intensity (Bits)')
            plt.yscale('linear')
            plt.title(network)
            plt.show()

        # Plot intermediate results if check is True
        if check:
            pos_freq = np.round(len(freq)/2).astype(int)
            
            N = len(index_calib) 
            if N % 2 !=0:
                N=N+1
            
            fig, axes = plt.subplots( int(N/2), 2,figsize=(12, 7))
            axes = axes.flatten()
            for i,ind_calib in enumerate(range(len(index_calib))):
                ax = axes[i] 

                ax = axes[int(ind_calib)]
                ax.plot(axe_time[index_calib[ind_calib]-3*Npt_PerStep:index_calib[ind_calib]+Npt_PerStep-1],
                        rh[index_calib[ind_calib]-3*Npt_PerStep:index_calib[ind_calib]+Npt_PerStep-1,pos_freq],
                        label='RH')
                ax.plot(axe_time[index_calib[ind_calib]-3*Npt_PerStep:index_calib[ind_calib]+Npt_PerStep-1],
                        lh[index_calib[ind_calib]-3*Npt_PerStep:index_calib[ind_calib]+Npt_PerStep-1,pos_freq],
                        label='LH')
                ax.axvline(x=axe_time[index_calib[ind_calib]-3*Npt_PerStep], color='b', linestyle='-.', alpha=0.5)
                ax.axvline(x=axe_time[index_calib[ind_calib]+Npt_PerStep-1], color='b', linestyle='-.', alpha=0.5)
                ax.set_xlabel('Time (hours)')
                ax.set_ylabel('Intensity (Bits)')
            fig.suptitle('Example of calibration sequence (original)')
            
            if N != len(index_calib):
                axes[-1].axis('off')  # remove any unsued plot
            fig.canvas.header_visible = False
            plt.tight_layout()
            plt.show()

        # 1.2 Initialize calibration intensity array
        nb_calib = len(index_calib)
        int_bits_calib = np.zeros((4, nb_calib, 2))

        # Calculate mean intensity within each calibration sub-band of each calibration period
        for ind_calib in range(nb_calib):
            for ind_network in range(2):
                if ind_network == 0:
                    int_net = np.median(rh[index_calib[ind_calib]-3*Npt_PerStep:int(index_calib[ind_calib] + Npt_PerStep - 1),:],axis=1)
                else:
                    int_net = np.median(lh[index_calib[ind_calib]-3*Npt_PerStep:int(index_calib[ind_calib] + Npt_PerStep - 1),:],axis=1)

                for ind in range(4):
                    int_bits_calib[ind, ind_calib, ind_network] = np.mean(int_net[int(ind*Npt_PerStep):int((ind+1)*(Npt_PerStep-1))]) 

        # Convert to dB
        int_db_input = np.array([20.51, 30.51, 40.51, 50.51])  # Reference levels in dB
        coeff_calib = np.zeros((2, 2))

        # Fit calibration coefficients
        for ind_network in range(2):
            uu = np.array([
            np.median(int_bits_calib[0, :, ind_network]),
            np.median(int_bits_calib[1, :, ind_network]),
            np.median(int_bits_calib[2, :, ind_network]),
            np.median(int_bits_calib[3, :, ind_network])
            ])
            xx = np.log10(uu)
            coeff_calib[:, ind_network] = np.polyfit(xx, int_db_input, 1)
            #print(f"Network {ind_network} / Coeff {coeff_calib} / XX {xx}")
        # Plot calibration coefficients if check is True
        if check:
            plt.figure(figsize=(10, 8))
            for ind_calib in range(4):
                plt.subplot(2, 2, int(ind_calib)+1)
                for ind_network in range(2):
                    plt.plot(np.log10(int_bits_calib[ind_calib, :, ind_network]),marker='+')
                    plt.axhline(y=xx[ind_calib])
                    #plt.axhline(y=coeff_calib[ind_calib,ind_network ])
                plt.title('SubCalib'+str(ind_calib))   
                plt.xlabel('Calibration Sequence #')
                plt.ylabel('BITS')
            plt.show()
            plt.tight_layout()

        # Apply (frequency independent) calibration - data are then in dB
        calibrated_r_db = coeff_calib[1, 0] + coeff_calib[0, 0] * np.log10(rh)
        calibrated_l_db = coeff_calib[1, 1] + coeff_calib[0, 1] * np.log10(lh)

        # Plot calibrated data if check is True
        if check:
            plt.figure(figsize=(12,6))
            plt.plot(nda_time[index_calib[0]-3*Npt_PerStep:int(index_calib[0]+Npt_PerStep-1)],
                    calibrated_l_db[index_calib[0]-3*Npt_PerStep:int(index_calib[0]+Npt_PerStep-1),pos_freq],
                    label='LH',marker='+')
            plt.plot(nda_time[index_calib[0]-3*Npt_PerStep:int(index_calib[0]+Npt_PerStep-1)],
                    calibrated_r_db[index_calib[0]-3*Npt_PerStep:int(index_calib[0]+Npt_PerStep-1),pos_freq],
                    label='RH')
            plt.xlabel('Time (hours)')
            plt.ylabel('Intensity (dB)')
            plt.legend()
            #fig.canvas.header_visible = False
            plt.title('Calibration sequence after calibration')
            plt.show()

        # Convert dB to Jansky
        T_ref = 290.0  # Ambient temperature in Kelvin
        calibrated_l_jy = np.zeros_like(calibrated_l_db)
        calibrated_r_jy = np.zeros_like(calibrated_r_db)

        for ind_freq in range(len(freq)):
            ae = 24.0 * (T_ref / freq[ind_freq])**2
            if ae > 3500.0:
                ae = 3500.0
            cjy = T_ref * (2 * 1.380649e-23) / ae / 1e-26  # Conversion factor to Jansky )

            calibrated_l_jy[:,ind_freq] = cjy * 10**(calibrated_l_db[:,ind_freq] / 10) # Jansky (1Jy = 1e-26 W/(m^2.Hz))
            calibrated_r_jy[:,ind_freq] = cjy * 10**(calibrated_r_db[:,ind_freq] / 10)

        # Plot final results if pplot is True
        if pplot:
            vmin = 1e-2 
            vmax = 10 # 1E5SFU
            fig, (ax1, ax2,ax3) = plt.subplots(3, 1, figsize=(12, 8),sharex=True,sharey=True)
            im1 = ax1.imshow(np.log(calibrated_l_jy.T/1E4) ,vmin=vmin,vmax=vmax,
                aspect='auto',extent=[nda_time.min(), nda_time.max(), freq.min(), freq.max()],
                cmap='viridis')
            ax1.set_title(f"Left Polarization {year}/{month}/{day} - {mode}")
            ax1.set_xlabel('Time (UT)')
            ax1.set_ylabel('Frequency (MHz)')
            fig.colorbar(im1, ax=ax1, label='log$_{10}$(SFU)')


            im2 = ax2.imshow(np.log(calibrated_r_jy.T/1E4),vmin=vmin,vmax=vmax,
                aspect='auto',extent=[nda_time.min(), nda_time.max(), freq.min(), freq.max()],
                cmap='viridis')
            ax2.set_title('Right Polarization')
            ax2.set_xlabel('Time (UT)')
            ax2.set_ylabel('Frequency (MHz)')
            fig.colorbar(im2, ax=ax2, label='log$_{10}$(SFU)')

            im3 = ax3.imshow(((calibrated_l_jy - calibrated_r_jy) /(calibrated_l_jy + calibrated_r_jy)).T,
                aspect='auto',extent=[nda_time.min(), nda_time.max(), freq.min(), freq.max()],
                cmap='seismic')
            ax3.set_title('(Left-Right)/(Left+Right)')
            ax3.set_xlabel('Time (UT)')
            ax3.set_ylabel('Frequency (MHz)')
            fig.colorbar(im3, ax=ax3, label='V/I')

            updating = False
        
            ax1.callbacks.connect('xlim_changed', lambda event: sli.on_xlims_change3axis(event, ax1, ax2, ax3,fig))
            ax1.callbacks.connect('ylim_changed', lambda event: sli.on_xlims_change3axis(event, ax1, ax2, ax3,fig))
            ax2.callbacks.connect('xlim_changed', lambda event: sli.on_xlims_change3axis(event, ax1, ax2, ax3,fig))
            ax2.callbacks.connect('ylim_changed', lambda event: sli.on_xlims_change3axis(event, ax1, ax2, ax3,fig))
            ax3.callbacks.connect('xlim_changed', lambda event: sli.on_xlims_change3axis(event, ax1, ax2, ax3,fig))
            ax3.callbacks.connect('ylim_changed', lambda event: sli.on_xlims_change3axis(event, ax1, ax2, ax3,fig))
            fig.canvas.header_visible = False
            plt.tight_layout()
            plt.show()
    else:
        print('No calibration sequence found. Return RAW data ')
        plt.figure(figsize=(12,5))
        plt.plot(axe_time,mean_int)
        plt.xlabel('Time (Hours)')
        plt.axhline(y=thres_calib, color='r', linestyle='--', label='Y = 23')
        
        plt.ylabel('Intensity (Bits)')
        plt.yscale('linear')
        plt.title(network)
        plt.show()

        calibrated_l_jy=lh
        calibrated_r_jy = rh

    return calibrated_l_jy, calibrated_r_jy, nda_time, freq
