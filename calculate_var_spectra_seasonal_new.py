# makes ADCP of spectra using HFR code
import sys
sys.path.append("../codes/")
import numpy as np
from matplotlib.dates import date2num, num2date, datestr2num, datetime
import matplotlib.pyplot as plt
from matplotlib import rcParams as rcp
import netCDF4 as S
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from slomo_processing import *

rcp['axes.labelsize'] = 12.
rcp['xtick.labelsize'] = 12.
rcp['ytick.labelsize'] = 12.

# interpolated data in time
tit = 'ADCPE'
lat = -4.669
lon =  55.646
file = S.Dataset('/Users/carolina/SLOMO/clean_data/adcp_E_slomo_hourly_20162019.nc', mode='r')
u = file.variables['u'][:]
v = file.variables['v'][:]
time = file.variables['time_vel'][:]
depth_e = file.variables['depth'][:]

f, b = coriolis(lat)
fday = f * 86400 / (2 * np.pi)

window = 'blackman'
ci = 95
nens = 6
PSD = np.ones((np.int(u.shape[0] / nens), u.shape[-1])) * 1e20
# make hourly data
for ii in range(0, u.shape[-1]):
	freq, PSD[:, ii], ehi, elo, dof = plot_spectrum(time, u[:, ii], v[:,ii], window, ci, nens)
	print(str(ii) + ' out of ' + str(u.shape[-1]))

fnegi = np.where(freq < 0)
fposi = np.where(freq >= 0)
psneg = np.squeeze(PSD[fnegi, :])
fneg = freq[fnegi]
fpos = freq[fposi]
pspos = np.squeeze(PSD[fposi, :])
N = len(PSD)
dt = (np.diff(time)).mean()
df = 1 / (N * dt)
vartot = np.sum(PSD, axis=0) * df

highi = 26.18
lowi = 24 / 4 # 168
xi = mpl.mlab.find((abs(freq) > lowi) & (abs(freq) < highi))
dff = np.diff(freq[xi])
var_hf = np.sum(PSD[xi, :], axis=0) * df
tot_hf = var_hf / vartot * 100

highi = 24 / 4 # 8
lowi = 24 / 57 # 168
xi = mpl.mlab.find((abs(freq) > lowi) & (abs(freq) < highi))
dff = np.diff(freq[xi])
var_tidal = np.sum(PSD[xi, :], axis=0) * df
tot_tidal = var_tidal / vartot * 100

highi = 24 / 57 # 8
lowi = 24 / 176 # 168
xi = mpl.mlab.find((abs(freq) > lowi) & (abs(freq) < highi))
dff = np.diff(freq[xi])
var_inertial = np.sum(PSD[xi, :], axis=0) * df
tot_inertial = var_inertial / vartot * 100

highi = 24 / 176 # 8
lowi = 0 #
xi = mpl.mlab.find((abs(freq) > lowi) & (abs(freq) < highi))
var_lf = np.sum(PSD[xi, :], axis=0) * df
tot_lf = var_lf / vartot * 100

fig = plt.figure(figsize=(10, 4.95))
gs = gridspec.GridSpec(1, 3, hspace=0.1, wspace=0.3)
plt.subplot(gs[0, 0:2])
ax = plt.gca()
ax.loglog(np.abs(fneg), psneg.mean(-1) * np.abs(fneg), color='k', linewidth = 1.8, alpha=0.9) #you can multiply by 24 to have cph like cedric
ax.loglog(np.abs(fpos), pspos.mean(-1) * np.abs(fpos), color='k', linewidth = 1.5, alpha=0.6)
leg2 = ax.legend(['clockwise','counterclockwise'], loc='lower left',prop={'size':12})
leg2._drawFrame=False
ax.set_xlabel('$cpd$')
ax.set_ylabel(r'variance preserving spectra $[m^2 s^{-2}]$')
erraxis = 1e-3
ax.axvline(1, color='steelblue', linestyle='dashed',lw=1)
ax.axvline(2, color='steelblue', linestyle='dashed',lw=1)
ax.text(0.65, 1.01, r'$K1$',
        transform=ax.transAxes, fontsize=14)
ax.text(0.75, 1.01, r'$M2$',
        transform=ax.transAxes, fontsize=14)
ax.axvline(np.abs(fday), color='steelblue', linestyle='dashed',lw=1)
ax.axvspan(0, 1 / 168 * 24, alpha=0.4, color='black')
ax.axvspan(1 / 168 * 24, 1 / 48 * 24, alpha=0.3, color='black')
ax.axvspan(1 / 48 * 24, 1 / 4 * 24, alpha=0.2, color='black')
ax.axvspan(1 / 4 * 24, 26.15 * 24, alpha=0, color='black')
conf_x = 0.02
conf_y0 = 0.03
conf = conf_y0 * dof / [ehi, elo]
ax.plot([conf_x, conf_x], conf, color='k', lw=1.5)
ax.plot(conf_x, conf_y0, color='k', linestyle='none',
        marker='o', ms=8, mew=2)
ax.text(.02+.005, erraxis * ehi, '95 %', fontsize=16)
ax.text(0.43, 1.01, r'$f$',
        transform=ax.transAxes, fontsize=14)
ax.text(0.023258247089575433, 0.2, 'lf', fontsize=12, bbox=dict(facecolor='white',
alpha=0.9))
ax.text(0.115, 0.2, 'near-inertial', fontsize=12, bbox=dict(facecolor='white',
alpha=0.9))
ax.text(1.6221254844587365, 0.2, 'tidal', fontsize=12, bbox=dict(facecolor='white',
alpha=0.9))
ax.text(9, 0.2, 'hf', fontsize=12, bbox=dict(facecolor='white',
alpha=0.9))
ax.text(0.05, 1.01, 'a)',
        transform=ax.transAxes, fontsize=16)
ax.axis([0.0050263297491194864,
 13.57690401148181,
 3.605515178223577e-05,
 0.38])

plt.subplot(gs[0, 2])
ax = plt.gca()
vars = [tot_lf.mean(), tot_inertial.mean(), tot_tidal.mean(), tot_hf.mean()]
plt.bar([1, 2, 3, 4], height=vars, color='steelblue', alpha=0.9)
ax.set_xticks([1, 2, 3, 4])
ax.text(0.05, 1.01, 'b)',
        transform=ax.transAxes, fontsize=16)
ax.set_xticklabels(('lf', 'near-','tidal', 'hf'), fontsize=12)
ax.text(1.4, -4, 'inertial', fontsize=12)
ax.set_ylabel('Percentage of total variance [%]')
# fig.suptitle(tit)
# plt.savefig('figures/var_hist_e'+tit+'.pdf', bbox_inches='tight')
