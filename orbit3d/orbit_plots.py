# Author: Yunlin Zeng
# Computes and plots possible orbits of companions to their host stars
# based on the orbital parameters from MCMC chains

import numpy as np
from random import randrange
from scipy.interpolate import interp1d
import scipy.optimize as op
import orbit
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.ticker import NullFormatter
from matplotlib.ticker import AutoMinorLocator
import sys





class OrbitPlots:

    def __init__(self, title, Hip, epoch_interval, cmref, RVfile=None, relAstfile=None, num_lines=50, pred_yr=[1990, 2000, 2010, 2020, 2030], cm_name='viridis', burnin=100, steps=1500, nplanets=1, rvpm_ref=0):

        self.title = title
        self.Hip = Hip
        self.epoch_interval = epoch_interval
        self.cmref = cmref
        self.num_lines = num_lines
        self.pred_yr = pred_yr
        if min(self.pred_yr) < min(epoch_interval) or max(pred_yr) > max(epoch_interval):
            raise ValueError('pred_yr should not exceed the range of epoch_interval')
        self.cm_name = cm_name
        self.cmlabel_dic = {'msec': r'$M_{comp} (M_\odot)$', 'ecc': 'eccentricity'}
        self.color_list = ['r', 'g', 'y', 'm', 'c', 'b']
        self.nplanets = nplanets
        self.rvpm_ref = rvpm_ref

        t = fits.open('HGCA_vDR2_corrected.fits')[1].data
        self.use_epoch_astrometry = False
        rvpath = '/Users/yunlin/physics/research/orbit_fitting/rvdata/' # '/Users/yunlin/physis/research/orbit_fitting/rvdata/'
        relAstpath = '/Users/yunlin/physics/research/orbit_fitting/relAstdata/'
        if RVfile != None:
            self.RVfile = rvpath + RVfile
        else:
            self.RVfile = rvpath + self.title + '_RV.dat'
        if relAstfile != None:
            self.relAstfile = relAstpath + relAstfile
        else:
            self.relAstfile = relAstpath + self.title + '_relAST.txt'

        self.steps = steps
        self.start_epoch = self.calendar_to_JD(self.epoch_interval[0])
        self.end_epoch = self.calendar_to_JD(self.epoch_interval[1])
        self.range_epoch = self.end_epoch - self.start_epoch
        self.epoch = np.linspace(self.start_epoch - 0.5*self.range_epoch, self.end_epoch + 0.5*self.range_epoch, self.steps)   # + 0.5*self.range_epoch in case some oribts have longer period and appear to be broken in astrometry / RV orbits
        self.epoch_cal = np.zeros(len(self.epoch))
        for i in range(len(self.epoch_cal)):
            self.epoch_cal[i] = self.JD_to_calendar(self.epoch[i])

        # load in mcmc's result
        path = '/Users/yunlin/physics/research/orbit_fitting/mcmc_chains/'
        file = self.title + '_chain000.fits'
        self.tt, self.lnp, self.extras = [fits.open(path + file)[i].data for i in range(3)]
        for i in range(nplanets):
            self.tt[:, :, 2+i*7+5] %= 2*np.pi
        self.beststep = np.where(self.lnp == self.lnp.max())

        # load in observed RV data
        rvdat = np.genfromtxt(self.RVfile)
        self.epoch_obs = rvdat[:, 0]     # it might be better to change this to epochself.RV_obs
        self.RV_obs = rvdat[:, 1]
        self.RV_obs_err = rvdat[:, 2]
        self.nRV = rvdat.shape[0]

        RVinst = (rvdat[:, 3]).astype(np.int32)
        # Check to see that the column we loaded was an integer
        assert np.all(RVinst == rvdat[:, 3])
        self.nInst = int(np.amax(rvdat[:, 3]) + 1)

        self.idx_dic = {}
        self.epoch_obs_dic = {}
        self.RV_obs_dic = {}
        self.RV_obs_err_dic = {}
        for i in range(self.nInst):
            self.idx_dic[i] = (np.where(RVinst == i)[0][0], np.where(RVinst == i)[0][-1])
            self.epoch_obs_dic[i] = self.epoch_obs[self.idx_dic[i][0]: self.idx_dic[i][-1] + 1]
            self.RV_obs_dic[i] = self.RV_obs[self.idx_dic[i][0]: self.idx_dic[i][-1] + 1]
            self.RV_obs_err_dic[i] = self.RV_obs_err[self.idx_dic[i][0]: self.idx_dic[i][-1] + 1]

        # load in reltive use_epoch_astrometry:
        try:
            reldat = np.genfromtxt(self.relAstfile, usecols=(1,2,3,4,5), skip_header=1)
            self.ep_relAst_obs = reldat[:, 0]
            for i in range(len(self.ep_relAst_obs)):
                self.ep_relAst_obs[i] = self.calendar_to_JD(self.ep_relAst_obs[i])
            # load in relative separations (arcsec)
            self.relsep_obs = reldat[:, 1]
            self.relsep_obs_err = reldat[:, 2]
            # load in position angles (degree)
            self.PA_obs = reldat[:, 3]
            self.PA_obs_err = reldat[:, 4]
            try:
                self.ast_planetID = (reldat[:, 5]).astype(np.int32)
            except:
                self.ast_planetID = (reldat[:, 0]*0).astype(np.int32)
            self.have_reldat = True
        except:
            self.have_reldat = False

        try:
            i = int(np.where(t['hip_id'] == Hip)[0])
            self.ep_pmra_obs = np.array([t['epoch_ra_hip'][i], t['epoch_ra_gaia'][i]])
            self.ep_pmdec_obs = np.array([t['epoch_dec_hip'][i], t['epoch_dec_gaia'][i]])
            self.pmra_obs = np.array([t['pmra_hip'][i], t['pmra_gaia'][i]])
            self.pmra_obs_err = np.array([t['pmra_hip_error'][i], t['pmra_gaia_error'][i]])
            self.pmdec_obs = np.array([t['pmdec_hip'][i], t['pmdec_gaia'][i]])
            self.pmdec_obs_err = np.array([t['pmdec_hip_error'][i], t['pmdec_gaia_error'][i]])
            for i in range(len(self.ep_pmra_obs)):
                self.ep_pmra_obs[i] = self.calendar_to_JD(self.ep_pmra_obs[i])
                self.ep_pmdec_obs[i] = self.calendar_to_JD(self.ep_pmdec_obs[i])
            self.have_pmdat = True
        except:
            self.have_pmdat = False

        ############### calculate RA, Dec, epoch and RV ###############

        # compute the orbital parameters for the most likely orbit
        data = orbit.Data(Hip, self.RVfile, self.relAstfile, self.use_epoch_astrometry)
        data.custom_epochs(self.epoch)
        model = orbit.Model(data)
        self.plx = data.plx      # parallax in arcsec
        self.dras_ml_list = []
        self.ddecs_ml_list = []
        self.relsep_ml_list = []
        self.PA_ml_list = []
        self.TA_ml_list = []
        self.omega_list = []

        for i in range(nplanets):
            par = orbit.Params(self.tt[self.beststep][0], i, nplanets)
            data.custom_epochs(self.epoch, iplanet=i)       # need custom_epochs again to assign ast_planetID for each point. If without the correct ast_planetID, orbit.calc_offsets will skip calculating the relative astrometry
            orbit.calc_EA_RPP(data, par, model)
            orbit.calc_RV(data, par, model)
            orbit.calc_offsets(data, par, model, i)

            self.dras_ml, self.ddecs_ml = model.return_dRA_dDec(par, data, model)
            self.dras_ml, self.ddecs_ml = -(1. + par.mpri/par.msec)*self.plx*self.dras_ml, -(1. + par.mpri/par.msec)*self.plx*self.ddecs_ml
            self.relsep_ml = model.return_relsep()*self.plx       # this is relative separation in terms of arcsec
            self.PA_ml = (model.return_PAs()*180/np.pi) % 360
            pmra_ml, pmdec_ml = model.return_proper_motions(par)
            pmra_ml, pmdec_ml = 1e3*self.plx*365.25*pmra_ml, 1e3*self.plx*365.25*pmdec_ml   # convert from arcsec/day to mas/yr
            if i == 0:
                self.pmra_ml = pmra_ml
                self.pmdec_ml = pmdec_ml
            else:
                self.pmra_ml += pmra_ml
                self.pmdec_ml += pmdec_ml
            self.dras_ml_list.append(self.dras_ml)
            self.ddecs_ml_list.append(self.ddecs_ml)
            self.relsep_ml_list.append(self.relsep_ml)
            self.PA_ml_list.append(self.PA_ml)
            self.TA_ml_list.append(model.return_TAs(par))
            self.omega_list.append(par.arg)

        self.RV_ml = model.return_RVs()
        self.pmra_ml += self.extras[self.beststep[0], self.beststep[1], 1]*1000
        self.pmdec_ml += self.extras[self.beststep[0], self.beststep[1], 2]*1000
        self.f_RVml = interp1d(self.epoch, self.RV_ml)

        # calculate the offset of the observed RV data

        '''
        calculate the differences of offsets (del_offset), and shift the data
        according to del_offset, which normalizes the offset for different instruments.
        '''
        offset_dic = {}
        del_offset_dic = {}
        print(self.nInst)
        for i in range(self.nInst):
            offset_dic[i] = self.extras[self.beststep][0][-self.nInst+i] # i=0 -> last one; i=1 -> last but one...
        offset = min(offset_dic.values())
        for i in range(self.nInst):
            del_offset_dic[i] = offset_dic[i] - offset
            self.RV_obs_dic[i] += del_offset_dic[i]

        # shift most likely RV to towards data points
        self.RV_ml -= offset

        # compute the orbital parameters for the randomly selected orbits
        # and shift up or down to line up with the data points
        midep_RV_obs = (self.epoch_obs[0] + self.epoch_obs[-1])/2 #self.epoch_obs[int(len(self.epoch_obs)/2)]
        RV_ref_val = self.f_RVml(midep_RV_obs)

        self.RV_dic = {}
        self.dras_dic = {}
        self.ddecs_dic = {}
        self.relsep_dic = {}
        self.PA_dic = {}
        self.pmra_dic = {}
        self.pmdec_dic = {}

        for j in range(self.num_lines):

            # get parameters from one single step of the mcmc chain
            walker_idx = randrange(self.tt.shape[0])
            step_idx = randrange(burnin, self.tt.shape[1])

            # clean up the initial model for each orbit
            model = orbit.Model(data)

            for i in range(nplanets):
                par = orbit.Params(self.tt[walker_idx, step_idx], i, nplanets)
                data.custom_epochs(self.epoch, iplanet=i)
                orbit.calc_EA_RPP(data, par, model)
                orbit.calc_RV(data, par, model)
                orbit.calc_offsets(data, par, model, i)

                dras, ddecs = model.return_dRA_dDec(par, data, model)
                dras, ddecs = -(1. + par.mpri/par.msec)*self.plx*dras, -(1. + par.mpri/par.msec)*self.plx*ddecs  # convert from AU to arcsec
                relsep = model.return_relsep()*self.plx
                PA = (model.return_PAs()*180/np.pi) % 360
                pmra_tmp, pmdec_tmp = model.return_proper_motions(par)
                pmra_tmp, pmdec_tmp = 1e3*self.plx*365.25*pmra_tmp, 1e3*self.plx*365.25*pmdec_tmp   # convert from arcsec/day to mas/yr
                if i == 0:
                    pmra = pmra_tmp
                    pmdec = pmdec_tmp
                else:
                    pmra += pmra_tmp
                    pmdec += pmdec_tmp

                cmref = getattr(par, self.cmref)
                self.dras_dic[(cmref, i)] = dras
                self.ddecs_dic[(cmref, i)] = ddecs
                self.relsep_dic[(cmref, i)] = relsep
                self.PA_dic[(cmref, i)] = PA

            RV = model.return_RVs()

            # line up each RV curve with respect to the most likely one
            f_RV = interp1d(self.epoch, RV)
            RV -= offset + (f_RV(midep_RV_obs) - RV_ref_val)
            # shift curves to the data points
            pmra += self.extras[walker_idx, step_idx, 1]*1000
            pmdec += self.extras[walker_idx, step_idx, 2]*1000

            self.RV_dic[cmref] = RV
            self.pmra_dic[cmref] = pmra
            self.pmdec_dic[cmref] = pmdec

        '''
        Sort the dras_dic, ddecs_dic, relsep_dic, PA_dic in terms of msec/ecc/etc.
        The keys of these four dictionaries are tuples, where the first argument is
        the value of msec/ecc/etc., and the second argument is iplanet (0, 1, 2...).
        After being sorted, each dictionary has num_lines*(nplanets+1) elements.
        The first num_lines elements are for the first companion, and their value of
        msec/ecc/etc. are from small to large. The second num_lines are for the second
        planet and so on.
        '''
        dics = [self.dras_dic, self.ddecs_dic, self.relsep_dic, self.PA_dic]
        for j in range(len(dics)):
            dic = dics[j]
            dic_tmp0 = {}
            for i in range(nplanets):
                dic_tmp1 = {}
                for key in dic:
                    if key[1] == i:
                        dic_tmp1[key] = dic[key]
                dic_tmp1 = dict(sorted(dic_tmp1.items(), key=lambda items: items[0][0]))
                dic_tmp0 = {**dic_tmp0, **dic_tmp1}
            dics[j] = dic_tmp0

        self.dras_dic = dics[0]
        self.ddecs_dic = dics[1]
        self.relsep_dic = dics[2]
        self.PA_dic = dics[3]

        self.RV_dic = dict(sorted(self.RV_dic.items(), key=lambda items: items[0]))
        self.pmra_dic = dict(sorted(self.pmra_dic.items(), key=lambda items: items[0]))
        self.pmdec_dic = dict(sorted(self.pmdec_dic.items(), key=lambda items: items[0]))

        self.dic_keys = []
        for key in self.dras_dic:
            self.dic_keys.append(key[0])
        self.RV_dic_vals = list(self.RV_dic.values())
        self.dras_dic_vals = list(self.dras_dic.values())
        self.ddecs_dic_vals = list(self.ddecs_dic.values())
        self.relsep_dic_vals = list(self.relsep_dic.values())
        self.PA_dic_vals = list(self.PA_dic.values())
        self.pmra_dic_vals = list(self.pmra_dic.values())
        self.pmdec_dic_vals = list(self.pmdec_dic.values())


    ############### Auxiliary Functions ###############

    # Julian date <-> calendar date conversion
    def JD_to_calendar(self, JD):
        a = int(JD + 0.5)
        if a < 2299161:
            c = a + 1524
        else:
            b = int((a - 1867216.25)/36524.25)
            c = a + b - int(b/4) + 1525
        d = int((c - 122.1)/365.25)
        e = int(365.25*d)
        f = int((c - e)/30.6001)

        D = c - e - int(30.6001*f) + ((JD + 0.5) - int(JD + 0.5))
        M = f - 1 - 12*int(f/14.)
        Y = d - 4715 - int((7 + M)/10.)
        year = Y + M/12. + D/365.25
        return year

    def calendar_to_JD(self, year, M=1, D=1):
        if M <= 2:
            y = year - 1
            m = M + 12
        else:
            y = year
            m = M
        if year <= 1583: # more precisely should be less than or equal to 10/4/1582
            B = -2
        else:
            B = int(y/400.) - int(y/100.)
        UT = 0
        JD = int(365.25*y) + int(30.6001*(m+1)) + B + 1720996.5  + D + UT/24.
        return JD


    ############### plot astrometric orbit ###############

    def astrometry(self):

        for j in range(self.nplanets):

            # setup the normalization and the colormap
            self.nValues = np.array(self.dic_keys[self.num_lines*j: self.num_lines*(j+1)])
            self.normalize = mcolors.Normalize(vmin=self.nValues.min(), vmax=self.nValues.max())
            self.colormap = getattr(cm, self.cm_name)

            # setup the colorbar
            self.sm = cm.ScalarMappable(norm=self.normalize, cmap=self.colormap)
            self.sm.set_array(self.nValues)

            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111)

            # plot the num_lines randomly selected curves
            for i in range(self.num_lines):
                ax.plot(self.dras_dic_vals[i+j*self.num_lines], self.ddecs_dic_vals[i+j*self.num_lines], color=self.colormap(self.normalize(self.nValues[i])), alpha=0.3)

            # plot the most likely ones
            dras_ml = self.dras_ml_list[j]
            ddecs_ml = self.ddecs_ml_list[j]
            ax.plot(dras_ml, ddecs_ml, color='black')

            # plot the relAst data points
            try:
                assert self.have_reldat == True
                for i in range(len(self.ep_relAst_obs)):
                    if self.ast_planetID[i] != j:
                        continue
                    ra_obs = self.relsep_obs[i] * np.sin(self.PA_obs[i]*np.pi/180)
                    dec_obs = self.relsep_obs[i] * np.cos(self.PA_obs[i]*np.pi/180)
                    ax.scatter(ra_obs, dec_obs, s=45, facecolors='coral', edgecolors='none', zorder=99)
                    ax.scatter(ra_obs, dec_obs, s=45, facecolors='none', edgecolors='k', zorder=100)
            except:
                pass

            # plot the 5 predicted positions of the companion from 1990 to 2030
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            x_range = x1 - x0
            y_range = y1 - y0
            epoch_int = []
            for year in self.epoch_cal:
                epoch_int.append(int(year))

            text_x_max = 0
            text_x_min = 0
            text_y_max = 0
            text_y_min = 0
            for year in self.pred_yr:
                idx = epoch_int.index(year)
                x = dras_ml[idx]
                y = ddecs_ml[idx]
                r = np.sqrt(x**2 + y**2)
                ax.scatter(x, y, s=55, facecolors='none', edgecolors='k', zorder=100)

                # avoid overlapping between text and plot
                if x >= 0 and y >= 0:
                    text_x = x*(r + 0.14*(x1 - x0))/r
                    text_y = y*(r + 0.03*(y1 - y0))/r
                    ax.text(text_x, text_y, str(year), fontsize=10)
                elif x >= 0 and y <= 0:
                    text_x = x*(r + 0.14*(x1 - x0))/r
                    text_y = y*(r + 0.05*(y1 - y0))/r
                    ax.text(text_x, text_y, str(year), fontsize=10)
                elif x <= 0 and y >= 0:
                    text_x = x*(r + 0.03*(x1 - x0))/r
                    text_y = y*(r + 0.03*(y1 - y0))/r
                    ax.text(text_x, text_y, str(year), fontsize=10)
                elif x <= 0 and y <= 0:
                    text_x = x*(r + 0.03*(x1 - x0))/r
                    text_y = y*(r + 0.06*(y1 - y0))/r
                    ax.text(text_x, text_y, str(year), fontsize=10)

                if text_x > text_x_max:
                    text_x_max = text_x
                if text_x < text_x_min:
                    text_x_min = text_x
                if text_y > text_y_max:
                    text_y_max = text_y
                if text_y < text_y_min:
                    text_y_min = text_y

            # avoid the text to exceed the frame
            if abs(text_x_min - x0) < 0.10*x_range:
                x0 -= 0.10*x_range
            if abs(text_x_max - x1) < 0.10*x_range:
                x1 += 0.10*x_range
            if abs(text_y_min - y0) < 0.05*y_range:
                y0 -= 0.05*y_range
            if abs(text_y_max - y1) < 0.05*y_range:
                y1 += 0.05*y_range

            # find the positions of nodes and periastron

            '''
            When par.arg (omega) is negative, it means the periastron is below the plane
            of the sky. We can set omega = -par.arg, which is the angle symmetric with
            respect to the plane of the sky. Although this agnle doesn't mean anything, the
            same algorithm below can be applied to locate the position of nodes.
            '''
            omega = abs(self.omega_list[j])
            TA_ml = self.TA_ml_list[j]
            idx_node0 = np.where(abs(TA_ml - (np.pi - omega)) == min(abs(TA_ml - (np.pi - omega))))[0]
            idx_node1 = np.where(abs(TA_ml - (-omega)) == min(abs(TA_ml - (-omega))))[0]
            node0 = (dras_ml[idx_node0], ddecs_ml[idx_node0])
            node1 = (dras_ml[idx_node1], ddecs_ml[idx_node1])
            idx_pras = np.where(abs(TA_ml) == min(abs(TA_ml)))[0]
            pras = (dras_ml[idx_pras], ddecs_ml[idx_pras])

            # plot line of nodes, periastron and the direction of motion of companion star, and label the host star
            ax.plot([node0[0], node1[0]], [node0[1], node1[1]], 'k--')
            ax.plot([0, pras[0]], [0, pras[1]], 'k:')
            arrow = mpatches.FancyArrowPatch((dras_ml[idx_pras][0], ddecs_ml[idx_pras][0]), (dras_ml[idx_pras+1][0], ddecs_ml[idx_pras+1][0]), arrowstyle='->', mutation_scale=25, zorder=100)
            ax.add_patch(arrow)
            ax.plot(0, 0, marker='*', color='black')

            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_aspect(abs((x1-x0)/(y1-y0)))
            fig.colorbar(self.sm, ax=ax, label=self.cmlabel_dic[self.cmref])
            ax.invert_xaxis()
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.tick_params(direction='in', which='both', left=True, right=True, bottom=True, top=True)
            ax.set_xlabel(r'$\Delta \alpha$ [arcsec]')
            ax.set_ylabel(r'$\Delta \delta$ [arcsec]')
            ax.set_title(self.title + ' Astrometric Orbits')

            fig.savefig('astrometric_orbit' + str(j) + '_' + self.title)


    ############### plot the RV orbits ###############

    def RV(self):

        # setup the normalization and the colormap
        self.nValues = np.array(self.dic_keys[self.num_lines*self.rvpm_ref: self.num_lines*(self.rvpm_ref+1)])
        self.normalize = mcolors.Normalize(vmin=self.nValues.min(), vmax=self.nValues.max())
        self.colormap = getattr(cm, self.cm_name)

        # setup the colorbar
        self.sm = cm.ScalarMappable(norm=self.normalize, cmap=self.colormap)
        self.sm.set_array(self.nValues)

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)

        # plot the num_lines randomly selected curves
        for i in range(self.num_lines):
            ax.plot(self.epoch, self.RV_dic_vals[i], color=self.colormap(self.normalize(self.nValues[i])), alpha=0.3)

        # plot the most likely one
        ax.plot(self.epoch, self.RV_ml, color='black')

        # plot the observed data points (RV & relAst)
        for i in range(self.nInst):
            ax.plot(self.epoch_obs_dic[i], self.RV_obs_dic[i], self.color_list[i]+'o', markersize=2)

        # manually change the x tick labels from JD to calendar years
        epoch_ticks = np.linspace(self.start_epoch, self.end_epoch, 5)
        epoch_label = np.zeros(len(epoch_ticks))
        for i in range(len(epoch_ticks)):
            epoch_label[i] = round(self.JD_to_calendar(epoch_ticks[i]))

        ax.set_xlim(self.start_epoch, self.end_epoch)
        ax.set_xticks(epoch_ticks)
        ax.set_xticklabels([str(int(i)) for i in epoch_label])
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_aspect((x1-x0)/(y1-y0))
        fig.colorbar(self.sm, ax=ax, label=self.cmlabel_dic[self.cmref])
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(direction='in', which='both', left=True, right=True, bottom=True, top=True)
        ax.set_xlabel('Date (yr)')
        # ax.set_xlabel('Julian Days')
        ax.set_ylabel('RV (m/s)')
        ax.set_title(self.title + ' RV Orbits')
        # ax.set_xlim(self.calendar_to_JD(1990), self.calendar_to_JD(1991))

        fig.savefig('RV_orbit_' + self.title)


    ############### plot the relative RV and O-C ###############

    def relRV(self):

        # setup the normalization and the colormap
        self.nValues = np.array(self.dic_keys[self.num_lines*self.rvpm_ref: self.num_lines*(self.rvpm_ref+1)])
        self.normalize = mcolors.Normalize(vmin=self.nValues.min(), vmax=self.nValues.max())
        self.colormap = getattr(cm, self.cm_name)

        # setup the colorbar
        self.sm = cm.ScalarMappable(norm=self.normalize, cmap=self.colormap)
        self.sm.set_array(self.nValues)

        fig = plt.figure(figsize=(5, 6))
        ax1 = fig.add_axes((0.15, 0.3, 0.8, 0.6))
        ax2 = fig.add_axes((0.15, 0.1, 0.8, 0.15))

        # plot the num_lines randomly selected curves
        self.f_RVml = interp1d(self.epoch, self.RV_ml)
        RV_OC = self.RV_dic_vals

        for i in range(self.num_lines):
            ax1.plot(self.epoch, self.RV_dic_vals[i], color=self.colormap(self.normalize(self.nValues[i])), alpha=0.3)
            for j in range(len(self.epoch)):
                RV_OC[i][j] -= self.f_RVml(self.epoch[j])
            ax2.plot(self.epoch, RV_OC[i], color=self.colormap(self.normalize(self.nValues[i])), alpha=0.3)

        # plot the most likely ones
        ax1.plot(self.epoch, self.RV_ml, color='black')
        ax2.plot(self.epoch, np.zeros(len(self.epoch)), 'k--', dashes=(5, 5))

        # plot the observed data points
        datOC_list = []
        for i in range(self.nInst):
            ax1.errorbar(self.epoch_obs_dic[i], self.RV_obs_dic[i], yerr=self.RV_obs_err_dic[i], fmt=self.color_list[i]+'o', ecolor='black', capsize=3, zorder=99)
            ax1.scatter(self.epoch_obs_dic[i], self.RV_obs_dic[i], s=45, facecolors='none', edgecolors='k', zorder=100, alpha=0.5)
            for j in range(len(self.epoch_obs_dic[i])):
                OC = self.RV_obs_dic[i][j] - self.f_RVml(self.epoch_obs_dic[i][j])
                datOC_list.append(OC)
                ax2.errorbar(self.epoch_obs_dic[i][j], OC, yerr=self.RV_obs_err_dic[i][j], fmt=self.color_list[i]+'o', ecolor='black', capsize=3, zorder=99)
                ax2.scatter(self.epoch_obs_dic[i][j], OC, s=45, facecolors='none', edgecolors='k', zorder=100, alpha=0.5)

        # manually change the x tick labels from JD to calendar years
        epoch_ticks = np.linspace(self.epoch_obs[0], self.epoch_obs[-1], 5)
        epoch_label = np.zeros(len(epoch_ticks))
        for i in range(len(epoch_ticks)):
            epoch_label[i] = round(self.JD_to_calendar(epoch_ticks[i]))

        range_ep_obs = max(self.epoch_obs) - min(self.epoch_obs)
        range_RV_obs = max(self.RV_obs) - min(self.RV_obs)
        ax1.set_xlim(min(self.epoch_obs) + range_ep_obs/20., max(self.epoch_obs) - range_ep_obs/20.)
        ax1.set_ylim(min(self.RV_obs) - range_RV_obs/10., max(self.RV_obs) + range_RV_obs/10.)
        ax1.xaxis.set_major_formatter(NullFormatter())
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax1.tick_params(direction='in', which='both', left=True, right=True, bottom=True, top=True)
        ax1.set_title(self.title)
        ax1.set_ylabel('Relative RV (m/s)')

        range_datOC = max(datOC_list) - min(datOC_list)
        ax2.set_xlim(min(self.epoch_obs) + range_ep_obs/20., max(self.epoch_obs) - range_ep_obs/20.)
        ax2.set_ylim(min(datOC_list) - range_datOC/5., max(datOC_list) + range_datOC/5.)
        ax2.set_xticks(epoch_ticks)
        ax2.set_xticklabels([str(int(i)) for i in epoch_label])
        ax2.xaxis.set_minor_locator(AutoMinorLocator())
        ax2.yaxis.set_minor_locator(AutoMinorLocator())
        ax2.tick_params(direction='in', which='both', left=True, right=True, bottom=True, top=True)
        ax2.set_xlabel('Epoch (yr)')
        ax2.set_ylabel('O-C')

        plt.savefig('relRV_OC_' + self.title)


    ############### plot the separation and O-C ###############

    def relsep(self):

        for k in range(self.nplanets):

            # setup the normalization and the colormap
            self.nValues = np.array(self.dic_keys[self.num_lines*k: self.num_lines*(k+1)])
            self.normalize = mcolors.Normalize(vmin=self.nValues.min(), vmax=self.nValues.max())
            self.colormap = getattr(cm, self.cm_name)

            # setup the colorbar
            self.sm = cm.ScalarMappable(norm=self.normalize, cmap=self.colormap)
            self.sm.set_array(self.nValues)

            try:
                assert self.have_reldat == True
                assert np.any(self.ast_planetID) == k

                fig = plt.figure(figsize=(5, 6))
                ax1 = fig.add_axes((0.15, 0.3, 0.8, 0.6))
                ax2 = fig.add_axes((0.15, 0.1, 0.8, 0.15))

                # plot the num_lines randomly selected curves
                relsep_OC = self.relsep_dic_vals[k*self.num_lines: (k+1)*self.num_lines]
                for i in range(self.num_lines):
                    ax1.plot(self.epoch, self.relsep_dic_vals[i+k*self.num_lines], color=self.colormap(self.normalize(self.nValues[i])), alpha=0.3)
                    for j in range(len(self.epoch)):
                        relsep_OC[i][j] -= self.relsep_ml_list[k][j]
                    ax2.plot(self.epoch, relsep_OC[i], color=self.colormap(self.normalize(self.nValues[i])), alpha=0.3)

                # plot the most likely ones
                ax1.plot(self.epoch, self.relsep_ml_list[k], color='black')
                ax2.plot(self.epoch, np.zeros(len(self.epoch)), 'k--', dashes=(5, 5))

                # plot the observed data points
                datOC_list = []
                for i in range(len(self.ep_relAst_obs)):
                    if self.ast_planetID[i] != k:
                        continue
                    ax1.errorbar(self.ep_relAst_obs[i], self.relsep_obs[i], yerr=self.relsep_obs_err[i], color='coral', fmt='o', ecolor='black', capsize=3, zorder=99)
                    ax1.scatter(self.ep_relAst_obs[i], self.relsep_obs[i], s=45, facecolors='none', edgecolors='k', zorder=100, alpha=1)

                    # compute the most likely value of relsep at the epoch when the data point is recorded
                    ep_obs = np.array([self.ep_relAst_obs[i]])
                    data = orbit.Data(self.Hip, self.RVfile, self.relAstfile, self.use_epoch_astrometry)
                    data.custom_epochs(ep_obs)
                    model = orbit.Model(data)
                    par = orbit.Params(self.tt[self.beststep][0], k, self.nplanets)
                    data.custom_epochs(ep_obs, iplanet=k)
                    orbit.calc_EA_RPP(data, par, model)
                    orbit.calc_offsets(data, par, model, k)
                    relsep = model.return_relsep()*self.plx

                    # find the difference between observed relsep and most likely relsep for each point and plot the result in the OC plot
                    dat_OC = self.relsep_obs[i] - relsep
                    datOC_list.append(dat_OC)
                    ax2.errorbar(self.ep_relAst_obs[i], dat_OC, yerr=self.relsep_obs_err[i], color='coral', fmt='o', ecolor='black', capsize=3, zorder=99)
                    ax2.scatter(self.ep_relAst_obs[i], dat_OC, s=45, facecolors='none', edgecolors='k', zorder=100, alpha=1)

                # manually change the x tick labels from JD to calendar years
                epoch_ticks = np.linspace(self.ep_relAst_obs[0], self.ep_relAst_obs[-1], 5)
                epoch_label = np.zeros(len(epoch_ticks))
                for i in range(len(epoch_ticks)):
                    epoch_label[i] = round(self.JD_to_calendar(epoch_ticks[i]))

                self.range_ep_relAst_obs = max(self.ep_relAst_obs) - min(self.ep_relAst_obs)
                range_relsep_obs = max(self.relsep_obs)  - min(self.relsep_obs)
                ax1.set_xlim(min(self.ep_relAst_obs) - self.range_ep_relAst_obs/8., max(self.ep_relAst_obs) + self.range_ep_relAst_obs/8.)
                ax1.set_ylim(min(self.relsep_obs) - range_relsep_obs/2., max(self.relsep_obs) + range_relsep_obs/2.)
                ax1.xaxis.set_major_formatter(NullFormatter())
                ax1.xaxis.set_minor_locator(AutoMinorLocator())
                ax1.yaxis.set_minor_locator(AutoMinorLocator())
                ax1.tick_params(direction='in', which='both', left=True, right=True, bottom=True, top=True)
                ax1.set_title(self.title)
                ax1.set_ylabel('separation (arcsec)')

                range_datOC = max(datOC_list) - min(datOC_list)
                ax2.set_xlim(min(self.ep_relAst_obs) - self.range_ep_relAst_obs/8., max(self.ep_relAst_obs) + self.range_ep_relAst_obs/8.)
                ax2.set_ylim(min(datOC_list) - range_datOC, max(datOC_list) + range_datOC)
                ax2.set_xticks(epoch_ticks)
                ax2.set_xticklabels([str(int(i)) for i in epoch_label])
                ax2.xaxis.set_minor_locator(AutoMinorLocator())
                ax2.yaxis.set_minor_locator(AutoMinorLocator())
                ax2.tick_params(direction='in', which='both', left=True, right=True, bottom=True, top=True)
                ax2.set_xlabel('Epoch (yr)')
                ax2.set_ylabel('O-C')

            except:
                fig = plt.figure(figsize=(5, 5))
                ax = fig.add_axes((0.15, 0.1, 0.8, 0.8))

                # plot the num_lines randomly selected curves
                for i in range(self.num_lines):
                    ax.plot(self.epoch, self.relsep_dic_vals[i+k*self.num_lines], color=self.colormap(self.normalize(self.nValues[i])), alpha=0.3)

                # plot the most likely one
                ax.plot(self.epoch, self.relsep_ml_list[k], color='black')

                # manually change the x tick labels from JD to calendar years
                epoch_ticks = np.linspace(self.start_epoch, self.end_epoch, 5)
                epoch_label = np.zeros(len(epoch_ticks))
                for i in range(len(epoch_ticks)):
                    epoch_label[i] = round(self.JD_to_calendar(epoch_ticks[i]))

                ax.set_xlim(self.start_epoch, self.end_epoch)
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                ax.set_xticks(epoch_ticks)
                ax.set_xticklabels([str(int(i)) for i in epoch_label])
                ax.tick_params(direction='in', which='both', left=True, right=True, bottom=True, top=True)
                ax.set_title(self.title)
                ax.set_xlabel('date (yr)')
                ax.set_ylabel('separation (arcsec)')

            plt.savefig('relsep_OC_' + str(k) + '_' + self.title)


    ############### plot the position angle and O-C ###############

    def PA(self):

        for k in range(self.nplanets):

            # setup the normalization and the colormap
            self.nValues = np.array(self.dic_keys[self.num_lines*k: self.num_lines*(k+1)])
            self.normalize = mcolors.Normalize(vmin=self.nValues.min(), vmax=self.nValues.max())
            self.colormap = getattr(cm, self.cm_name)

            # setup the colorbar
            self.sm = cm.ScalarMappable(norm=self.normalize, cmap=self.colormap)
            self.sm.set_array(self.nValues)

            try:
                assert self.have_reldat == True
                assert np.any(self.ast_planetID) == k
                fig = plt.figure(figsize=(5, 6))
                ax1 = fig.add_axes((0.15, 0.3, 0.8, 0.6))
                ax2 = fig.add_axes((0.15, 0.1, 0.8, 0.15))

                # plot the num_lines randomly selected curves
                f_PAml = interp1d(self.epoch, self.PA_ml_list[k])
                PA_OC = self.PA_dic_vals[k*self.num_lines: (k+1)*self.num_lines]

                for i in range(self.num_lines):
                    ax1.plot(self.epoch, self.PA_dic_vals[i+k*self.num_lines], color=self.colormap(self.normalize(self.nValues[i])), alpha=0.3)
                    for j in range(len(self.epoch)):
                        PA_OC[i][j] -= f_PAml(self.epoch[j])
                    ax2.plot(self.epoch, PA_OC[i], color=self.colormap(self.normalize(self.nValues[i])), alpha=0.3)

                # plot the most likely ones
                ax1.plot(self.epoch, self.PA_ml_list[k], color='black')
                ax2.plot(self.epoch, np.zeros(len(self.epoch)), 'k--', dashes=(5, 5))

                # plot the observed data points
                datOC_list = []
                for i in range(len(self.ep_relAst_obs)):
                    if self.ast_planetID[i] != k:
                        continue
                    ax1.errorbar(self.ep_relAst_obs[i], self.PA_obs[i], yerr=self.PA_obs_err[i], color='coral', fmt='o', ecolor='black', capsize=3, zorder=99)
                    ax1.scatter(self.ep_relAst_obs[i], self.PA_obs[i], s=45, facecolors='none', edgecolors='k', zorder=100, alpha=1)
                    dat_OC = self.PA_obs[i] - f_PAml(self.ep_relAst_obs[i])
                    datOC_list.append(dat_OC)
                    ax2.errorbar(self.ep_relAst_obs[i], dat_OC, yerr=self.PA_obs_err[i], color='coral', fmt='o', ecolor='black', capsize=3, zorder=99)
                    ax2.scatter(self.ep_relAst_obs[i], dat_OC, s=45, facecolors='none', edgecolors='k', zorder=100, alpha=1)

                # manually change the x tick labels from JD to calendar years
                epoch_ticks = np.linspace(self.ep_relAst_obs[0], self.ep_relAst_obs[-1], 5)
                epoch_label = np.zeros(len(epoch_ticks))
                for i in range(len(epoch_ticks)):
                    epoch_label[i] = round(self.JD_to_calendar(epoch_ticks[i]))

                self.range_ep_relAst_obs = max(self.ep_relAst_obs) - min(self.ep_relAst_obs)
                range_PA_obs = max(self.PA_obs)  - min(self.PA_obs)
                ax1.set_xlim(min(self.ep_relAst_obs) - self.range_ep_relAst_obs/8., max(self.ep_relAst_obs) + self.range_ep_relAst_obs/8.)
                ax1.set_ylim(min(self.PA_obs) - range_PA_obs/5., max(self.PA_obs) + range_PA_obs/5.)
                ax1.xaxis.set_major_formatter(NullFormatter())
                ax1.xaxis.set_minor_locator(AutoMinorLocator())
                ax1.yaxis.set_minor_locator(AutoMinorLocator())
                ax1.tick_params(direction='in', which='both', left=True, right=True, bottom=True, top=True)
                ax1.set_title(self.title)
                ax1.set_ylabel(r'Position Angle ($^{\circ}$)')

                range_datOC = max(datOC_list) - min(datOC_list)
                ax2.set_xlim(min(self.ep_relAst_obs) - self.range_ep_relAst_obs/8., max(self.ep_relAst_obs) + self.range_ep_relAst_obs/8.)
                ax2.set_ylim(min(datOC_list) - range_datOC, max(datOC_list) + range_datOC)
                ax2.set_xticks(epoch_ticks)
                ax2.set_xticklabels([str(int(i)) for i in epoch_label])
                ax2.xaxis.set_minor_locator(AutoMinorLocator())
                ax2.yaxis.set_minor_locator(AutoMinorLocator())
                ax2.tick_params(direction='in', which='both', left=True, right=True, bottom=True, top=True)
                ax2.set_xlabel('Epoch (yr)')
                ax2.set_ylabel('O-C')

            except:
                fig = plt.figure(figsize=(5, 5))
                ax = fig.add_axes((0.15, 0.1, 0.8, 0.8))

                # plot the num_lines randomly selected curves
                for i in range(self.num_lines):
                    ax.plot(self.epoch, self.PA_dic_vals[i+k*self.num_lines], color=self.colormap(self.normalize(self.nValues[i])), alpha=0.3)

                # plot the most likely one
                ax.plot(self.epoch, self.PA_ml_list[k], color='black')

                # manually change the x tick labels from JD to calendar years
                epoch_ticks = np.linspace(self.start_epoch, self.end_epoch, 5)
                epoch_label = np.zeros(len(epoch_ticks))
                for i in range(len(epoch_ticks)):
                    epoch_label[i] = round(self.JD_to_calendar(epoch_ticks[i]))

                ax.set_xlim(self.start_epoch, self.end_epoch)
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                ax.set_xticks(epoch_ticks)
                ax.set_xticklabels([str(int(i)) for i in epoch_label])
                ax.tick_params(direction='in', which='both', left=True, right=True, bottom=True, top=True)
                ax.set_title(self.title)
                ax.set_xlabel('date (yr)')
                ax.set_ylabel(r'Position Angle ($^{\circ}$)')

            plt.savefig('PA_OC_' + str(k) + '_' + self.title)


    ############### plot the proper motion and O-C ###############

    def properMotion(self):

        # setup the normalization and the colormap
        self.nValues = np.array(self.dic_keys[self.num_lines*self.rvpm_ref: self.num_lines*(self.rvpm_ref+1)])
        self.normalize = mcolors.Normalize(vmin=self.nValues.min(), vmax=self.nValues.max())
        self.colormap = getattr(cm, self.cm_name)

        # setup the colorbar
        self.sm = cm.ScalarMappable(norm=self.normalize, cmap=self.colormap)
        self.sm.set_array(self.nValues)

        try:
            assert self.have_pmdat == True
            fig = plt.figure(figsize=(11, 6))
            ax1 = fig.add_axes((0.10, 0.30, 0.35, 0.60))
            ax2 = fig.add_axes((0.10, 0.10, 0.35, 0.15))
            ax3 = fig.add_axes((0.55, 0.30, 0.35, 0.60))
            ax4 = fig.add_axes((0.55, 0.10, 0.35, 0.15))

            # plot the num_lines randomly selected curves

            '''
            set the curves (connected by self.steps points) in the OC plot equal
            to the regular proper motion curves and each points on the curves
            will be subtracted by the most likely value later, in order to make
            the OC plots. The randomly selected regular proper motion curves
            will also be plotted here.
            '''
            pmra_OC = self.pmra_dic_vals
            pmdec_OC = self.pmdec_dic_vals
            for i in range(self.num_lines):
                ax1.plot(self.epoch, self.pmra_dic_vals[i], color=self.colormap(self.normalize(self.nValues[i])), alpha=0.3)
                ax3.plot(self.epoch, self.pmdec_dic_vals[i], color=self.colormap(self.normalize(self.nValues[i])), alpha=0.3)
                for j in range(len(self.epoch)):
                    pmra_OC[i][j] -= self.pmra_ml[j]
                    pmdec_OC[i][j] -= self.pmdec_ml[j]
                ax2.plot(self.epoch, pmra_OC[i], color=self.colormap(self.normalize(self.nValues[i])), alpha=0.3)
                ax4.plot(self.epoch, pmdec_OC[i], color=self.colormap(self.normalize(self.nValues[i])), alpha=0.3)

            # plot the most likely ones
            ax1.plot(self.epoch, self.pmra_ml, color='black')
            ax2.plot(self.epoch, np.zeros(len(self.epoch)), 'k--', dashes=(5, 5))
            ax3.plot(self.epoch, self.pmdec_ml, color='black')
            ax4.plot(self.epoch, np.zeros(len(self.epoch)), 'k--', dashes=(5, 5))

            # plot the observed data points
            pmradatOC_list = []
            pmdecdatOC_list = []
            ax1.errorbar(self.ep_pmra_obs, self.pmra_obs, yerr=self.pmra_obs_err, color='coral', fmt='o', ecolor='black', capsize=3, zorder=99)
            ax1.scatter(self.ep_pmra_obs, self.pmra_obs, s=45, facecolors='none', edgecolors='k', zorder=100, alpha=1)
            ax3.errorbar(self.ep_pmdec_obs, self.pmdec_obs, yerr=self.pmra_obs_err, color='coral', fmt='o', ecolor='black', capsize=3, zorder=99)
            ax3.scatter(self.ep_pmdec_obs, self.pmdec_obs, s=45, facecolors='none', edgecolors='k', zorder=100, alpha=1)

            # the OC plots
            for i in range(len(self.ep_pmra_obs)):
                ep_obs = np.array([self.ep_pmdec_obs[i]])

                # compute the most likely value of pmra, pmdec at the epoch when the data point is recorded
                for l in range(self.nplanets):
                    data = orbit.Data(self.Hip, self.RVfile, self.relAstfile, self.use_epoch_astrometry)
                    data.custom_epochs(ep_obs)
                    model = orbit.Model(data)
                    par = orbit.Params(self.tt[self.beststep][0], l, self.nplanets)
                    data.custom_epochs(ep_obs, iplanet=l)
                    orbit.calc_EA_RPP(data, par, model)
                    orbit.calc_offsets(data, par, model, l)
                    pmra_tmp, pmdec_tmp = model.return_proper_motions(par)
                    pmra_tmp, pmdec_tmp = 1e3*self.plx*365.25*pmra_tmp, 1e3*self.plx*365.25*pmdec_tmp   # convert from arcsec/day to mas/yr
                    if l == 0:
                        pmra = pmra_tmp
                        pmdec = pmdec_tmp
                    else:
                        pmra += pmra_tmp
                        pmdec += pmdec_tmp

                pmra += self.extras[self.beststep[0], self.beststep[1], 1]*1000
                pmdec += self.extras[self.beststep[0], self.beststep[1], 2]*1000
                pmra_OC = self.pmra_obs[i] - pmra
                pmdec_OC = self.pmdec_obs[i] - pmdec
                pmradatOC_list.append(pmra_OC)
                pmdecdatOC_list.append(pmdec_OC)

            ax2.errorbar(self.ep_pmra_obs, pmradatOC_list, yerr=self.pmra_obs_err, color='coral', fmt='o', ecolor='black', capsize=3, zorder=99)
            ax2.scatter(self.ep_pmra_obs, pmradatOC_list, s=45, facecolors='none', edgecolors='k', zorder=100, alpha=1)
            ax4.errorbar(self.ep_pmdec_obs, pmdecdatOC_list, yerr=self.pmdec_obs_err, color='coral', fmt='o', ecolor='black', capsize=3, zorder=99)
            ax4.scatter(self.ep_pmdec_obs, pmdecdatOC_list, s=45, facecolors='none', edgecolors='k', zorder=100, alpha=1)

            # manually change the x tick labels from JD to calendar years
            epoch_ticks = np.linspace(self.ep_pmra_obs[0], self.ep_pmra_obs[-1], 5)
            epoch_label = np.zeros(len(epoch_ticks))
            for i in range(len(epoch_ticks)):
                epoch_label[i] = round(self.JD_to_calendar(epoch_ticks[i]))

            self.range_eppm_obs = max(self.ep_pmra_obs) - min(self.ep_pmra_obs)
            range_pmra_obs = max(self.pmra_obs)  - min(self.pmra_obs)
            ax1.set_xlim(min(self.ep_pmra_obs) - self.range_eppm_obs/5., max(self.ep_pmra_obs) + self.range_eppm_obs/5.)
            ax1.set_ylim(min(self.pmra_obs) - range_pmra_obs/5., max(self.pmra_obs) + range_pmra_obs/5.)
            ax1.xaxis.set_major_formatter(NullFormatter())
            ax1.xaxis.set_minor_locator(AutoMinorLocator())
            ax1.yaxis.set_minor_locator(AutoMinorLocator())
            ax1.tick_params(direction='in', which='both', left=True, right=True, bottom=True, top=True)
            ax1.set_title(self.title)
            ax1.set_ylabel(r'$\Delta \mu_{\alpha}$ (mas/yr)')

            range_pmdec_obs = max(self.pmdec_obs)  - min(self.pmdec_obs)
            ax3.set_ylabel(r'$\Delta \mu_{\alpha}$ mas/yr')
            ax3.set_xlim(min(self.ep_pmdec_obs) - self.range_eppm_obs/5., max(self.ep_pmdec_obs) + self.range_eppm_obs/5.)
            ax3.set_ylim(min(self.pmdec_obs) - range_pmdec_obs/5., max(self.pmdec_obs) + range_pmdec_obs/5.)
            ax3.xaxis.set_major_formatter(NullFormatter())
            ax3.xaxis.set_minor_locator(AutoMinorLocator())
            ax3.yaxis.set_minor_locator(AutoMinorLocator())
            ax3.tick_params(direction='in', which='both', left=True, right=True, bottom=True, top=True)
            ax3.set_title(self.title)
            ax3.set_ylabel(r'$\Delta \mu_{\delta}$ (mas/yr)')

            range_pmradatOC = max(pmradatOC_list) - min(pmradatOC_list)
            ax2.set_xlim(min(self.ep_pmra_obs) - self.range_eppm_obs/5., max(self.ep_pmra_obs) + self.range_eppm_obs/5.)
            ax2.set_ylim(min(pmradatOC_list) - range_pmradatOC, max(pmradatOC_list) + range_pmradatOC)
            ax2.set_xticks(epoch_ticks)
            ax2.set_xticklabels([str(int(i)) for i in epoch_label])
            ax2.xaxis.set_minor_locator(AutoMinorLocator())
            ax2.yaxis.set_minor_locator(AutoMinorLocator())
            ax2.tick_params(direction='in', which='both', left=True, right=True, bottom=True, top=True)
            ax2.set_xlabel('Epoch (yr)')
            ax2.set_ylabel('O-C')

            range_pmdecdatOC = max(pmdecdatOC_list) - min(pmdecdatOC_list)
            ax4.set_xlim(min(self.ep_pmdec_obs) - self.range_eppm_obs/5., max(self.ep_pmdec_obs) + self.range_eppm_obs/5.)
            ax4.set_ylim(min(pmdecdatOC_list) - range_pmdecdatOC, max(pmdecdatOC_list) + range_pmdecdatOC)
            ax4.set_xticks(epoch_ticks)
            ax4.set_xticklabels([str(int(i)) for i in epoch_label])
            ax4.xaxis.set_minor_locator(AutoMinorLocator())
            ax4.yaxis.set_minor_locator(AutoMinorLocator())
            ax4.tick_params(direction='in', which='both', left=True, right=True, bottom=True, top=True)
            ax4.set_xlabel('Epoch (yr)')
            ax4.set_ylabel('O-C')

        except:
            fig = plt.figure(figsize=(11, 5))
            ax1 = fig.add_axes((0.10, 0.1, 0.35, 0.77))
            ax2 = fig.add_axes((0.60, 0.1, 0.35, 0.77))

            # plot the num_lines randomly selected curves
            for i in range(self.num_lines):
                ax1.plot(self.epoch, self.pmra_dic_vals[i], color=self.colormap(self.normalize(self.nValues[i])), alpha=0.3)
                ax2.plot(self.epoch, self.pmdec_dic_vals[i], color=self.colormap(self.normalize(self.nValues[i])), alpha=0.3)

            # plot the most likely ones
            ax1.plot(self.epoch, self.pmra_ml, color='black')
            ax2.plot(self.epoch, self.pmdec_ml, color='black')

            # manually change the x tick labels from JD to calendar years
            epoch_ticks = np.linspace(self.start_epoch, self.end_epoch, 5)
            epoch_label = np.zeros(len(epoch_ticks))
            for i in range(len(epoch_ticks)):
                epoch_label[i] = round(self.JD_to_calendar(epoch_ticks[i]))

            ax1.set_xlim(self.start_epoch, self.end_epoch)
            ax1.xaxis.set_minor_locator(AutoMinorLocator())
            ax1.yaxis.set_minor_locator(AutoMinorLocator())
            ax1.set_xticks(epoch_ticks)
            ax1.set_xticklabels([str(int(i)) for i in epoch_label])
            ax1.tick_params(direction='in', which='both', left=True, right=True, bottom=True, top=True)
            ax1.set_title(self.title)
            ax1.set_xlabel('date (yr)')
            ax1.set_ylabel(r'$\Delta \mu_{\alpha}$ (mas/yr)')

            ax2.set_xlim(self.start_epoch, self.end_epoch)
            ax2.xaxis.set_minor_locator(AutoMinorLocator())
            ax2.yaxis.set_minor_locator(AutoMinorLocator())
            ax2.set_xticks(epoch_ticks)
            ax2.set_xticklabels([str(int(i)) for i in epoch_label])
            ax2.tick_params(direction='in', which='both', left=True, right=True, bottom=True, top=True)
            ax2.set_title(self.title)
            ax2.set_xlabel('date (yr)')
            ax2.set_ylabel(r'$\Delta \mu_{\delta}$ (mas/yr)')

        plt.savefig('pm_OC_' + self.title)


# OPs = OrbitPlots('Gl758', 95319, (1900, 2100), 'msec')
# OPs = OrbitPlots('HD159062', 85653, (1990, 2250), 'msec')
# OPs = OrbitPlots('HD4747', 3850, (1990, 2030), 'msec')
# OPs = OrbitPlots('HD92987', 52472, (1990, 2030), 'msec')
# OPs = OrbitPlots('HD45701', 30480, (1990, 2030), 'msec')
# OPs = OrbitPlots('Gl86', 10138, (1960, 2050), 'msec', RVfile='HD13445_AAT_accelcorrected.vels', burnin=500, steps=15000, nplanets=2)
# OPs.astrometry()
# OPs.RV()
# OPs.relRV()
# OPs.relsep()
# OPs.PA()
# OPs.properMotion()

plt.show()


# some changes to this version of plot_orbits.py:
# change variable name epochrA_obs -> ep_relAst_obs
# change variable name range_eprA_obs -> range_ep_relAst_obs
# change variable name epoch_JD (which was a typo) -> epoch_cal
# delete - 0.1*self.range_epoch in self.epoch
# in def chi_sqr f_RVml -> f_ml, RV_obs -> data_obs, RV_obs_err -> data_obs_err
# in the function argument add nplanet
# PA / PA_ml = use %360
# generalize the computation part to multi-companion case
# change the way of sorting the dictionary so that it is applicable for either single companion or multiple companions
# no longer use similar triangle to plot the observed astrometric data points but use relsep*sin/cos(PA) directly
# compute the values instead of interpolation
# use extras[-1] as RVzero instead of using chi_sqr to calculate





































# end
