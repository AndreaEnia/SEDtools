# Written by A. Enia, Nov2020, yes, through the fucking pandemic.

import numpy as np
import re
import astropy.units as u

class Sed3FitOutput(object):
    """
    potato potato potato
    """
    
    def __init__(self, fitfilename, sedfilename):
        """
        potato potato potato
        """
        fitfile = open(fitfilename)
        sedfile = open(sedfilename)
        fitinfo = fitfile.readlines()
        sedinfo = sedfile.readlines()
        fitfile.close()
        sedfile.close()
        
        #strip out newline characters
        for i in range(len(fitinfo)):
            fitinfo[i] = fitinfo[i].strip()
        for i in range(len(sedinfo)):
            sedinfo[i] = sedinfo[i].strip()
        
        #first go through fitinfo
        filternames = fitinfo[1].strip("#")
        filternames = filternames.split()
        flux = np.array(fitinfo[2].split(), dtype=float) * u.Jy
        fluxerr = np.array(fitinfo[3].split(), dtype=float) * u.Jy
        predicted = np.array(fitinfo[23].split(), dtype=float) * u.Jy
        self.obs_filters = filternames
        self.obs_flux = flux 
        self.obs_flux_err = fluxerr
        self.obs_predict = predicted
        
        bestfitmodel = fitinfo[8].split()
        self.bestfit_i_sfh = int(bestfitmodel[0])
        self.bestfit_i_ir = int(bestfitmodel[1])
        self.bestfit_i_torus = float(bestfitmodel[2])
        self.bestfit_chi2 = float(bestfitmodel[3])
        self.bestfit_redshift = float(bestfitmodel[4])
        
        bestfitparams = fitinfo[10].strip('.#') + fitinfo[13].strip('.#')
        bestfitparams = re.split('\.+',bestfitparams)[:-4]
        bestfitresults = list(map(float,fitinfo[11].split())) + list(map(float,fitinfo[14].split()))
        self.bestfitparams_names = [self.clean_param_names(i) for i in bestfitparams]
        assert len(bestfitparams) == len(bestfitresults)
        for i,paramname in enumerate(bestfitparams):
            setattr(self,self.clean_param_names(paramname),bestfitresults[i])
 
        #now the different AGN fractions (in different ranges)
        agnfrac_results = list(map(float,fitinfo[20].split()))
        agnfrac_ranges = []
        temp = re.split('\.+', fitinfo[19].strip('.#'))   
        agnfrac_ranges.append(temp[0]+'.'+temp[1]+'.'+temp[2])
        agnfrac_ranges.append(temp[3]+'.'+temp[4]+'.'+temp[5])
        [agnfrac_ranges.append(temp[i]) for i in [-4, -3, -2, -1]]
        assert len(agnfrac_ranges) == len(agnfrac_results)
        self.agn_fractions = {}
        for rangename, value in zip(agnfrac_ranges, agnfrac_results): self.agn_fractions[rangename] = value

        #now working on the marginal PDF histograms for each parameter
        marginalpdfs = fitinfo[26:]
        #first, need to split the pdfs into each parameter
        self.marginal_pdfs = {}
        self.marginal_percentiles = {}
        hash_idx = []
        for i in range(len(marginalpdfs)):
            if '#' in marginalpdfs[i]:
                hash_idx.append(i)
        assert len(hash_idx) % 2 == 0
        for i in range(len(hash_idx)//2):
            param = marginalpdfs[hash_idx[2*i]].strip(' #.')
            marginal = marginalpdfs[hash_idx[2*i]+1:hash_idx[2*i+1]]
            marginal = np.array([j.split() for j in marginal],dtype=float)
            percentile = np.array(marginalpdfs[hash_idx[2*i+1]+1].split(),dtype=float)
            self.marginal_pdfs[self.clean_param_names(param)] = marginal
            self.marginal_percentiles[self.clean_param_names(param)] = percentile
            
        #now time for the SED file
        self.sed_model_params = {}
        #there are model names and params on lines 2 & 3 and 5 & 6
        modelparams = sedinfo[2].strip('.#')
        modelparams = re.split('\.+',modelparams)
        model_vals = list(map(float,sedinfo[3].split()))
        assert len(modelparams) == len(model_vals)
        for i,paramname in enumerate(modelparams):
            self.sed_model_params[self.clean_param_names(paramname)] = model_vals[i]
        modelparams = sedinfo[5].strip('.#')
        modelparams = re.split('\.+',modelparams)
        model_vals = list(map(float,sedinfo[6].split()))
        assert len(modelparams) == len(model_vals)
        for i,paramname in enumerate(modelparams):
            self.sed_model_params[self.clean_param_names(paramname)] = model_vals[i]
        #sed is from line 10 to the end. 
        #seven columns, log lambda, Total, Tot_no_AGN, Stars_Unattenuated, Torus, Stars_attenuated, Dusty_SF:
        model_sed = sedinfo[10:]
        model_sed = [i.split() for i in model_sed]
        self.sed_model = np.array(model_sed,dtype=float)
        self.sed_model_logwaves = self.sed_model[:,0] * u.dex(u.AA)
        self.sed_model_log_total = self.sed_model[:,1] * u.dex(u.Lsun / u.AA)
        self.sed_model_log_total_noAGN = self.sed_model[:,2] * u.dex(u.Lsun / u.AA)
        self.sed_model_log_stars_unattenuated = self.sed_model[:,3] * u.dex(u.Lsun / u.AA)
        self.sed_model_log_torus = self.sed_model[:,4] * u.dex(u.Lsun / u.AA)
        self.sed_model_log_stars_attenuated = self.sed_model[:,5] * u.dex(u.Lsun / u.AA)
        self.sed_model_log_dusty_SF = self.sed_model[:,6] * u.dex(u.Lsun / u.AA)

    @staticmethod
    def clean_param_names(paramname):
        """
        this removes the character '()/^*' from param names
        """
        paramname = paramname.replace('(','')
        paramname = paramname.replace(')','')
        paramname = paramname.replace('/','_')
        paramname = paramname.replace('^','_')
        paramname = paramname.replace('*','star')
        return paramname
