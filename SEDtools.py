import pandas as pd
import warnings
warnings.filterwarnings("ignore")

class SEDProperties:
    def __init__(self, sed_path, fit_path, z_type, code, cosmology = 'Planck15'):
        import os, sys
        import full_magphys_read, sed3fit_read

        import numpy as np
        from astropy import units as u
        from astropy.constants import c as v_lux
        from astropy.constants import L_sun
        if cosmology == 'Planck15': from astropy.cosmology import Planck15 as cosmo
        elif cosmology == 'FlatLambdaCDM':
            from astropy.cosmology import FlatLambdaCDM
            cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3)  
                    
        if code == 'magphys':
            self.results = full_magphys_read.MagphysOutput(fit_path, sed_path, z_type)
            self.z_source = self.results.bestfit_redshift
            self.z_type = z_type
            self.DL = cosmo.luminosity_distance(self.z_source).to('m')
            self.filters = self.results.obs_filters
            self.lambda_observed = ([Wvlghts_dictionary[band_disambiguation(i)] for i in self.filters]*u.micrometer).to('Angstrom')
            if z_type == 'phot':
                self.SED_observed = self.results.obs_flux # Already in Jy
                self.SED_observed_errors = self.results.obs_flux_err # Already in Jy
                self.SED_predicted = self.results.obs_predict # Already in Jy
                self.lambda_model = self.results.sed_model_logwaves # Already in AA
                self.SED_model = 10**(self.results.sed_model[:,1])*u.Jy # Already in Jy
                self.SED_model_StUn = 10**(self.results.sed_model[:,2])*u.Jy # Already in Jy
            elif z_type == 'spec':
                self.SED_observed = ((1+self.z_source)*1E26*(self.results.obs_flux.value*L_sun)/(4*np.pi*self.DL**2)).value*u.Jy
                self.SED_observed_errors = ((1+self.z_source)*1E26*(self.results.obs_flux_err.value*L_sun)/(4*np.pi*self.DL**2)).value*u.Jy
                self.SED_predicted = ((1+self.z_source)*1E26*(self.results.obs_predict.value*L_sun)/(4*np.pi*self.DL**2)).value*u.Jy
                self.lambda_model = self.results.sed_model_logwaves.to('Angstrom')
                y = 10**self.results.sed_model_logluminosity_lambda.value*(self.results.sed_model_logwaves.to('Angstrom')**2/v_lux.to('Angstrom/s'))
                self.SED_model = ((y/(4*np.pi*self.DL**2))*1E26*L_sun).value*u.Jy
                y = 10**self.results.sed_model[:,2]*(self.results.sed_model_logwaves.to('Angstrom')**2/v_lux.to('Angstrom/s'))
                self.SED_model_StUn = ((y/(4*np.pi*self.DL**2))*1E26*L_sun).value*u.Jy
            self.SED_model_dustySF = self.SED_model - self.SED_model_StUn
        elif code == 'sed3fit':
            self.results = sed3fit_read.Sed3FitOutput(fit_path, sed_path)
            self.z_source = self.results.bestfit_redshift
            self.z_type = z_type
            self.DL = cosmo.luminosity_distance(self.z_source).to('m')
            self.filters = self.results.obs_filters
            self.lambda_observed = ([Wvlghts_dictionary[band_disambiguation(i)] for i in self.filters]*u.micrometer).to('Angstrom')
            self.SED_observed = ((1+self.z_source)*1E26*(self.results.obs_flux.value*L_sun)/(4*np.pi*self.DL**2)).value*u.Jy
            self.SED_observed_errors = ((1+self.z_source)*1E26*(self.results.obs_flux_err.value*L_sun)/(4*np.pi*self.DL**2)).value*u.Jy
            self.SED_predicted = ((1+self.z_source)*1E26*(self.results.obs_predict.value*L_sun)/(4*np.pi*self.DL**2)).value*u.Jy
            self.lambda_model = self.results.sed_model_logwaves.to('Angstrom')
            y = 10**self.results.sed_model_log_total.value*(self.results.sed_model_logwaves.to('Angstrom')**2/v_lux.to('Angstrom/s'))
            self.SED_model = ((y/(4*np.pi*self.DL**2))*1E26*L_sun).value*u.Jy
            y = 10**self.results.sed_model_log_stars_unattenuated.value*(self.results.sed_model_logwaves.to('Angstrom')**2/v_lux.to('Angstrom/s'))
            self.SED_model_StUn = ((y/(4*np.pi*self.DL**2))*1E26*L_sun).value*u.Jy
            y = 10**self.results.sed_model_log_torus.value*(self.results.sed_model_logwaves.to('Angstrom')**2/v_lux.to('Angstrom/s'))
            self.SED_model_torus = ((y/(4*np.pi*self.DL**2))*1E26*L_sun).value*u.Jy
            y = 10**self.results.sed_model_log_dusty_SF.value*(self.results.sed_model_logwaves.to('Angstrom')**2/v_lux.to('Angstrom/s'))
            self.SED_model_dustySF = ((y/(4*np.pi*self.DL**2))*1E26*L_sun).value*u.Jy
  
        self.GOOD_UpLim = np.where(np.logical_and(self.SED_observed <= 0, self.SED_observed_errors > 0))[0]
        self.GOOD_Measure = np.where(~np.logical_and(self.SED_observed <= 0, self.SED_observed_errors > 0))[0]

    def plot_SED(self, ax_SED, identifier, \
                 plot_torus = False, plot_dustySF = False, plot_StUn = False, \
                 plot_uplims = True, plot_identifier = True, plot_chi = True, plot_RF = True, \
                 zphot_text_fontsize = 15, zphot_label_fontsize = 13, z_inset_xlength = 0.45, \
                 zphot_inset_xmin = 0, zphot_inset_xmax = 8, zphot_text_xtext = False, fontsize = 13, \
                 SED_observed_color = 'tab:red', SED_model_color = 'k', obs_ms = 5, \
                 xmin = 1E-1, xmax = 3E5, ymin = -8, ymax = 2):
        
        from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

        ax_SED.plot(self.lambda_model.to('micrometer'), np.log10(self.SED_model.value), color = SED_model_color, linewidth = 4.0)
        if plot_torus:
            try: ax_SED.plot(self.lambda_model.to('micrometer'), np.log10(self.SED_model_torus.value), color = 'tab:orange', linestyle = '-.', linewidth = 1.0)
            except: pass
        if plot_dustySF: ax_SED.plot(self.lambda_model.to('micrometer'), np.log10(self.SED_model_dustySF.value), color = 'tab:red', linestyle = '--', linewidth = 1.0)
        if plot_StUn: ax_SED.plot(self.lambda_model.to('micrometer'), np.log10(self.SED_model_StUn.value), color = 'tab:blue', linestyle = '--', linewidth = 1.0)
        x = self.lambda_observed.to('micrometer').value
        y = np.log10(self.SED_observed.value)
        yerr_log = np.array([LogError(t, terr)[0] if LogError(t, terr)[0] < 3 else LogError_v0(t, terr)[0] \
                             for t, terr in zip(self.SED_observed.value, self.SED_observed_errors.value)]) # The mean
        # Measurements
        ax_SED.errorbar(x[self.GOOD_Measure], y[self.GOOD_Measure], yerr=yerr_log[self.GOOD_Measure], \
                        color = SED_observed_color, fmt='o', mec = 'k', ms=obs_ms, capsize=3, zorder = 3)
        if plot_uplims:
            # Upper limits
            ax_SED.errorbar(x[self.GOOD_UpLim], np.log10(self.SED_observed_errors.value)[self.GOOD_UpLim], \
                            yerr=0.2, color='k', fmt='none', ms=15, zorder=3, capsize=3, uplims=True)
        # Scales and whatever
        ax_SED.set_xscale('log')
        #ax_SED.set_xlabel(r'Wavelength ($\mu$m)', size = 15), ax_SED.set_ylabel(r'$\log$ Flux (Jy)', size = 15)
        ax_SED.set_xlabel(r'$\log_{10} (\lambda/\mu{\rm m})$', size = 15), ax_SED.set_ylabel(r'$\log_{10} (f_{\lambda}\,/\,{\rm Jy})$', size = 15)
        x_text, y_text = np.percentile((xmin, xmax), 10), np.percentile((ymin, ymax), 3)
        if plot_identifier: ax_SED.plot(-100, -100, color = 'None', ms = 0, linewidth = 0, label = '{}'.format(identifier))
        if plot_chi: ax_SED.plot(-100, -100, color = 'None', ms = 0, linewidth = 0, label = r'$\chi^2 = {:.3f}$'.format(self.results.bestfit_chi2))
        if self.z_type == 'spec': ax_SED.plot(-100, -100, color = 'None', ms = 0, linewidth = 0, label = r'$z_s = {:.4f}$'.format(self.z_source))
        ax_SED.set_xlim(xmin, xmax), ax_SED.set_ylim(ymin, ymax)
        ax_SED.legend(loc = 'upper right', frameon=False, fontsize = fontsize)
        ax_SED.tick_params(axis='both', which='major', direction = 'in', labelsize=15)
        ax_SED.tick_params(axis='both', which='minor', direction = 'in', labelsize=8)
        ax_SED.yaxis.set_minor_locator(AutoMinorLocator())
       
        if plot_RF:
            ax_RF = ax_SED.twiny()
            RF_lambda = self.lambda_model.to('micrometer')*(1+self.z_source)
            #ax_RF.set_xlabel(r'Wavelength$_{\rm RF}$ ($\mu$m)', size = 15)
            ax_RF.set_xlabel(r'$\log_{10} (\lambda_{\rm RF}/\mu{\rm m})$', size = 15)
            ax_RF.set_xlim(xmin/(1+self.z_source), xmax/(1+self.z_source))
            ax_RF.set_xscale('log')
            ax_RF.tick_params(axis='both', which='major', direction = 'in', labelsize=15)
            ax_RF.tick_params(axis='both', which='minor', direction = 'in', labelsize=8)
            
        if self.z_type == 'phot':
            ax_zPDF = ax_SED.inset_axes([0.01, 0.60, z_inset_xlength, 0.37])
            z_bins = self.results.marginal_pdfs['redshift'][:,0]
            z_PDF = self.results.marginal_pdfs['redshift'][:,1]
            ax_zPDF.bar(z_bins, z_PDF, width = z_bins[1]-z_bins[0], edgecolor = SED_model_color, color = 'None')
            bestfit = self.results.marginal_percentiles['redshift'][2]
            m1s, p1s = bestfit-self.results.marginal_percentiles['redshift'][1], self.results.marginal_percentiles['redshift'][3]-bestfit
            text = r'$\mathcal{{Z}}_{{\rm phot}} = {0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$'.format(bestfit, p1s, m1s)
            if zphot_text_xtext != False: xtext = np.copy(zphot_text_xtext)
            else: xtext = (zphot_inset_xmax - zphot_inset_xmin)/2
            ytext = z_PDF.ptp()/1.4
            ax_zPDF.set_xlim(zphot_inset_xmin, zphot_inset_xmax)
            ax_zPDF.text(xtext, ytext, text, fontsize = zphot_text_fontsize, color = SED_model_color)
            ax_zPDF.set_xlabel(r'$\mathcal{Z}_{\rm PDF}$', fontsize = zphot_label_fontsize)
            ax_zPDF.xaxis.set_minor_locator(AutoMinorLocator())
            ax_zPDF.set_yticks([])
            ax_zPDF.set_yticks([], minor=True)
        else: pass
        return
    
    def plot_PDZ(self, ax_zPDF, identifier, \
                 zphot_text_fontsize = 15, zphot_label_fontsize = 13, z_inset_xlength = 0.45, \
                 zphot_inset_xmin = 0, zphot_inset_xmax = 8, zphot_text_xtext = False, fontsize = 13, \
                 SED_model_color = 'k', obs_ms = 5):
        
        from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
        z_bins = self.results.marginal_pdfs['redshift'][:,0]
        z_PDF = self.results.marginal_pdfs['redshift'][:,1]
        ax_zPDF.bar(z_bins, z_PDF, width = z_bins[1]-z_bins[0], edgecolor = SED_model_color, color = 'None')
        bestfit = self.results.marginal_percentiles['redshift'][2]
        m1s, p1s = bestfit-self.results.marginal_percentiles['redshift'][1], self.results.marginal_percentiles['redshift'][3]-bestfit
        text = r'$\mathcal{{Z}}_{{\rm phot}} = {0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$'.format(bestfit, p1s, m1s)
        if zphot_text_xtext != False: xtext = np.copy(zphot_text_xtext)
        else: xtext = (zphot_inset_xmax - zphot_inset_xmin)/2
        ytext = z_PDF.ptp()/1.4
        ax_zPDF.plot(xtext, ytext, color = 'None', ms = 0, linewidth = 0, label = '{}'.format(identifier))
        ax_zPDF.plot(xtext, ytext, color = 'None', ms = 0, linewidth = 0, label = '{}'.format(text))
        #ax_zPDF.text(xtext, ytext, text, fontsize = zphot_text_fontsize, color = SED_model_color)
        ax_zPDF.set_xlim(zphot_inset_xmin, zphot_inset_xmax)
        ax_zPDF.legend(loc = 'upper right', frameon=False, fontsize = fontsize)
        ax_zPDF.set_xlabel(r'$\mathcal{Z}_{\rm PDF}$', fontsize = zphot_label_fontsize)
        ax_zPDF.xaxis.set_minor_locator(AutoMinorLocator())
        ax_zPDF.yaxis.set_minor_locator(AutoMinorLocator())
        #ax_zPDF.set_yticks([], minor=True)
        return
    
    def PDZ_at_z(self, given_z):
        from scipy.interpolate import interp1d
        z_bins = self.results.marginal_pdfs['redshift'][:,0]
        z_PDF = self.results.marginal_pdfs['redshift'][:,1]
        x = np.linspace(0, 15, 1000)
        y = np.interp(x, z_bins, z_PDF, left=0, right=0)
        f = interp1d(x, y, bounds_error = False, fill_value = 0)
        return f(given_z)
        
import numpy as np

def round_sig(x, sig=2):
    return round(x, sig-((np.floor(np.log10(abs(x))))-1).astype('int'))
def round_arr(x, sig=2):
    return np.array([round_sig(el, sig) for el in x])

def band_disambiguation(band):
    if band in ['GALEX_FUV', 'GALEX FUV']: return 'GALEXFUV'
    if band in ['GALEX_NUV', 'GALEX NUV']: return 'GALEXNUV'
    if band in ['SUBARU_B']: return 'B'
    if band in ['SUBARU_V']: return 'V'
    if band in ['SDSS_u', 'SDSS u', 'Sloan u']: return 'u'
    if band in ['SDSS_g', 'SDSS g', 'Sloan g']: return 'g'
    if band in ['SDSS_r', 'SDSS r', 'Sloan r']: return 'r'
    if band in ['SDSS_i', 'SDSS i', 'Sloan i']: return 'i'
    if band in ['SDSS_z', 'SDSS z', 'Sloan z']: return 'z'
    if band in ['ACS_F435W']: return 'F435W'
    if band in ['ACS_F606W']: return 'F606W'
    if band in ['ACS_F775W']: return 'F775W'
    if band in ['ACS_F814W']: return 'F814W'
    if band in ['ACS_F850LP']: return 'F850LP'
    if band in ['WFC3_F105W']: return 'F105W'
    if band in ['WFC3_F125W']: return 'F125W'
    if band in ['WFC3_F140W']: return 'F140W'
    if band in ['WFC3_F160W']: return 'F160W'
    if band in ['VISTA_Y']: return 'Y'
    if band in ['2MASS_J', '2MASS J', 'VISTA_J']: return 'J'
    if band in ['2MASS_H', '2MASS H', 'VISTA_H']: return 'H'
    if band in ['2MASS_Ks', '2MASS Ks', '2MASS K', 'VISTA_Ks', 'KS']: return 'Ks'
    if band in ['WISE_3.4', 'WISE 3.4']: return 'WISEW1'
    if band in ['WISE_4.6', 'WISE 4.6']: return 'WISEW2'
    if band in ['WISE_12', 'WISE 12']: return 'WISEW3'
    if band in ['WISE_22', 'WISE 22']: return 'WISEW4'
    if band in ['IRAC3.6', 'Spitzer_3.6', 'Spitzer 3.6', 'IRAC_CH1_SCANDELS', 'CH1']: return 'IRAC1'
    if band in ['IRAC4.5', 'Spitzer_4.5', 'Spitzer 4.5', 'IRAC_CH2_SCANDELS', 'CH2']: return 'IRAC2'
    if band in ['IRAC5.8', 'Spitzer_5.8', 'Spitzer 5.8', 'IRAC_CH3', 'CH3']: return 'IRAC3'
    if band in ['IRAC8.0', 'Spitzer_8.0', 'Spitzer 8.0', 'IRAC_CH4', 'CH4']: return 'IRAC4'
    if band in ['Spitzer_16', 'Spitzer 16', '16', 'Spitzer/IRS/PUI', 'IRS/PUI', 'IRS', 'PUI']: return 'IRS16'
    if band in ['Spitzer_24', 'Spitzer 24', '24']: return 'MIPS24'
    if band in ['Spitzer_70', 'Spitzer 70', '70']: return 'MIPS70'
    if band in ['Spitzer_160', 'Spitzer 160', '160']: return 'MIPS160'
    if band in ['PACS_70', 'PACS 70']: return 'PACS70'
    if band in ['PACS_100', 'PACS 100', '100']: return 'PACS100'
    if band in ['PACS_160', 'PACS 160', '160']: return 'PACS160'
    if band in ['SPIRE_250', 'SPIRE 250', '250']: return 'SPIRE250'
    if band in ['SPIRE_350', 'SPIRE 350', '350']: return 'SPIRE350'
    if band in ['SPIRE_500', 'SPIRE 500', '500']: return 'SPIRE500'
    if band in ['SMA', 'SMA_880']: return 'SMA880'
    if band in ['SCUBA', 'SCUBA_850', 'SCUBA2', '850']: return 'SCUBA850'
    if band in ['AzTEC', 'MAMBO', 'MAMBO250', 'MAMBO250.', '1160']: return 'AzTEC+MAMBO'
    if band in ['VLA10cm', 'VLA3GHz', '3GHz']: return '10cm'
    if band in ['MIGHTEE', 'VLA20cm', 'VLA_L', 'VLA1.4GHz', '1_4GHz']: return '20cm'
    return band

def flatten(l, ltypes=(list, tuple)):
    ltype = type(l)
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1
    return ltype(l)

def LogError_v0(value, error):
    value, error = np.array(value), np.array(error)
    frac = 1.0 + (error/value)
    error_up = value * frac
    error_down = value / frac
    log_error_up = np.abs( np.log10(error_up) - np.log10(value) )
    log_error_down = np.abs( np.log10(value) - np.log10(error_down) )
    return log_error_up, log_error_down, 0.5*(log_error_up+log_error_down)

def LogError(value, error):
    rel_err = (np.array(error)/np.array(value))/np.log(10)
    return rel_err, rel_err, rel_err

def remove_duplicates(x):
    return np.array(list(dict.fromkeys(x)))

Wvlghts_dictionary = {'GALEX_FUV': 0.1528, 'GALEXFUV': 0.1528, 'GALEX_NUV': 0.2271, 'GALEXNUV': 0.2271, 'u': 0.3551, 'KPNO_U': 0.3593, 'LBC_U': 0.3633, \
                      'F275W': 0.2750, 'F336W': 0.3360, 'F350LP': 0.3500, 'F110W': 1.100, \
                    'F435W': 0.4318, 'B': 0.4458, 'V': 0.5477, 'SUBARU_ip': 0.7683, 'Y': 1.021419, 'g': 0.4686, \
                    'F606W': 0.5915, 'r': 0.6166, 'i': 0.748, 'F775W': 0.7693, 'F814W': 0.81, 'z': 0.8932, \
                    'F182M': 1.82, 'F210M': 2.10, 'F444W': 4.44, \
                    'F850LP': 0.9, 'F850L': 0.9, 'F105W': 1.01, 'F125W': 1.25, 'J': 1.25, '2MASS_J': 1.25, 'F140W': 1.39, 'F160W': 1.54, 'H': 1.65, '2MASS_H': 1.65,
                    'MOIRCS_K': 2.13, 'Ks': 2.16, 'K': 2.16, '2MASS_Ks': 2.16, 'CFHT_Ks': 2.16, 'WISE_3.4': 3.4, 'WISEW1': 3.4, 'Spitzer_3.6': 3.56, 
                    'IRAC1': 3.56, 'Spitzer_4.5': 4.51, 'IRAC2': 4.51, 'WISE_4.6': 4.6, 'WISEW2': 4.6, 'Spitzer_5.8': 5.76, 'IRAC3': 5.76, 'Spitzer_8.0': 8.0, 
                    'IRAC4': 8.0, 'WISE_12': 12.0, 'WISEW3': 12.0, 'IRS16': 16.0, 'WISE_22': 22.0, 'WISEW4': 22.0, 'Spitzer_24': 24.0, 'MIPS_24': 24.0, 
                    'MIPS24': 24.0, 'MIPS_70': 70.0, 'MIPS70': 70.0, 'PACS_70': 70.0, 'PACS70': 70.0, 'PACS_100': 100.0, 'PACS100': 100.0, 'PACS_160': 160.0, \
                    'PACS160': 160.0, 'SPIRE_250': 250.0, 'SPIRE250': 250.0, 'SPIRE_350': 350.0, 'SPIRE350': 350.0, 'SPIRE_500': 500.0, 'SPIRE500': 500.0, \
                    'SCUBA850': 850.0, 'SMA': 860.0, 'SMA880': 880.0, 'AzTEC+MAMBO': 1160.0, '10cm': 100000.0, '20cm': 200000.0, \
                    'NOEMAB4': 925.2, 'NOEMAB3': 1300.0, 'NOEMAB2': 2000.0, 'NOEMAB1': 3380.0, \
                    'A857um': 857, 'A860um': 860, 'A873um': 873, 'A874um': 874, 'A894um': 894, \
                    'A1209um': 1209, 'A1215um': 1215, 'A1218um': 1218, 'A1249um': 1249, 'A1287um': 1287, 'A1313um': 1313, \
                    'A2196um': 2196, 'A2899um': 2899, 'A2912um': 2912, 'A3010um': 3010, 'A3018um': 3018, 'A3151um': 3151, \
                    'A3187um': 3187, 'A3198um': 3198, 'A3214um': 3214, 'A3217um': 3217}