# band pass ADCP data
import sys
sys.path.append("../../codes/")
import numpy as np
from matplotlib import rcParams as rcp
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.dates import date2num, num2date, datestr2num, datetime
import netCDF4 as S
# external codes needed
sys.path.append('/Users/carolina/SLOMO/codes/python-seawater/')
import seawater as sw
import cm_xml_to_matplotlib as cm_xml
import cmocean
from slomo_processing import *

lin = 40
mycmap = cm_xml.make_cmap('blue-red-sat-outer.xml')

rcp['font.family'] = 'sans serif'
rcp['font.size'] = 12.
rcp['axes.labelsize'] = 16.
rcp['xtick.labelsize'] = 16.
rcp['ytick.labelsize'] = 16.
rcp['lines.linewidth'] = 2.
rcp['xtick.top'] = True
rcp['xtick.labeltop'] = False
rcp['xtick.bottom'] = True
rcp['xtick.labelbottom'] = True
# rcp['font.family'] = 'serif'

def centerify(text, width=-1):
    lines = text.split('\n')
    width = max(map(len, lines)) if width == -1 else width
    return '\n'.join(line.center(width) for line in lines)

lat = -4.669
lowi = 20 * 24
p = np.asarray([4.96, 24.16])

# load BV freq from ../TCH/read_SBE.py
mat = np.load('../../TCH/bf_E.npz')
time_tch = mat['time']
bf = mat['bf']

# Calculate BF from TOP and Bottom
file = S.Dataset('/Users/carolina/SLOMO/clean_data/SBE_37_E_slomo_20162019.nc', mode='r')
temp_b = file.variables['temp_bottom']
temp_t = file.variables['temp_top']
salt_b = file.variables['salt_bottom']
salt_t = file.variables['salt_top']

file = S.Dataset('/Users/carolina/SLOMO/clean_data/tch_SBE56_E_slomo_20162019.nc', mode='r')
temp_e = file.variables['temp'][:]
depth_e = file.variables['depth'][:]
time_e = file.variables['time'][:]
temp_final = np.ma.masked_array(np.ones((2, time_e.shape[0])))
salt_final = np.ma.masked_array(np.ones((2, time_e.shape[0])) * 35.25)
temp_final[0] = temp_e[:, -1]
temp_final[1] = temp_e[:, 0]
bf_temp, junk, junk = sw.geostrophic.bfrq(salt_final, temp_final,
	                                   p[:, None], lat=lat)
bf_temp = np.ma.masked_array(bf_temp[0])
bf_temp[temp_final[0].mask] = np.ma.masked
bf_temp_e = make_band_pass_1d(bf_temp, time_e, lowi, 'None', 'low')

file = S.Dataset('/Users/carolina/SLOMO/clean_data/tch_SBE56_W_slomo_20162019.nc', mode='r')
depth_w = file.variables['depth'][:]
temp_w = file.variables['temp'][:]
time_w = file.variables['time'][:]
temp_final = np.ma.masked_array(np.ones((2, time_w.shape[0])))
salt_final = np.ma.masked_array(np.ones((2, time_w.shape[0])) * 35.25)
temp_final[0] = temp_w[:, -1]
temp_final[1] = temp_w[:, 0]
bf_temp, junk, junk = sw.geostrophic.bfrq(salt_final, temp_final,
	                                   p[:, None], lat=lat)
bf_temp = np.ma.masked_array(bf_temp[0])
bf_temp[temp_final[0].mask] = np.ma.masked
bf_temp_w = make_band_pass_1d(bf_temp, time_w, lowi, 'None', 'low')

lowi = 30 * 24
highi = 70 * 24
temp_bp = make_band_pass_1d(bf_temp, time_w, lowi, highi, 'band')

fi1 = date2num(datetime.datetime.strptime('2017-05-19', '%Y-%m-%d'))
fi2 = date2num(datetime.datetime.strptime('2018-01-09', '%Y-%m-%d'))
temp_w_sh = temp_w[find_nearest(time_w, fi1)[1]: find_nearest(time_w, fi2)[1], :]
time_w_sh = time_w[find_nearest(time_w, fi1)[1]: find_nearest(time_w, fi2)[1]]

fi1 = date2num(datetime.datetime.strptime('2016-01', '%Y-%m'))
fi3 = date2num(datetime.datetime.strptime('2017-01', '%Y-%m'))
fi4 = date2num(datetime.datetime.strptime('2018-01', '%Y-%m'))
fi5 = date2num(datetime.datetime.strptime('2018-07', '%Y-%m'))
fi6 = date2num(datetime.datetime.strptime('2019-01', '%Y-%m'))

# depth is above instrument so we have to subtract
depth_new = depth_e - (28.77 + np.median(np.diff(depth_e))) # this is because contourf does not exactly plots in the point
li = 20
fig, axs = plt.subplots(3, 1, sharey = False, sharex=True, figsize=(14.48,6))
ax = axs[0]
fi = datetime.datetime.strptime("01-03-2016", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-06-2016", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-09-2016", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-12-2016", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.8)
fi = datetime.datetime.strptime("01-03-2017", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-06-2017", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-09-2017", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-12-2017", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-03-2018", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-06-2018", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-09-2018", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-12-2018", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-03-2019", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.5)

ag=0.9
ax.plot(num2date(time_e), temp_e.mean(1),
        alpha=0.8, color='k', lw=0.9, label='TchE')
ax.plot(num2date(time_w),
      temp_w.mean(1), color='darkorange',lw=0.8, alpha=ag, label='TchW')
ax.text(0.02, 0.83, 'a)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.9), color='k', fontsize=14)
ax.format_xdata = mpl.dates.DateFormatter('%b')
ax.axvline(x = fi3, color='k', linestyle='-',lw=2)
ax.axvline(x = fi4, color='k', linestyle='-',lw=2)
ax.axvline(x = fi6, color='k', linestyle='-',lw=2)
# ax.set_xlim(fi1, time_5e[-1])
ax.set_ylim(22, 32)
ax.set_ylabel(r'$[^oC]$')
ax.set_rasterized(True)
ax2 = axs[1]
fi = datetime.datetime.strptime("01-03-2016", "%d-%m-%Y")
ax2.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-06-2016", "%d-%m-%Y")
ax2.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-09-2016", "%d-%m-%Y")
ax2.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-12-2016", "%d-%m-%Y")
ax2.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-03-2017", "%d-%m-%Y")
ax2.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-06-2017", "%d-%m-%Y")
ax2.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-09-2017", "%d-%m-%Y")
ax2.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-12-2017", "%d-%m-%Y")
ax2.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-03-2018", "%d-%m-%Y")
ax2.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-06-2018", "%d-%m-%Y")
ax2.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-09-2018", "%d-%m-%Y")
ax2.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-12-2018", "%d-%m-%Y")
ax2.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-03-2019", "%d-%m-%Y")
ax2.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.5)



bf1 = np.ma.masked_greater(bf[0], .001)
ax2.plot(time_tch[0], bf1, lw=1.5, linestyle='-', alpha=0.6, color='k')
ax2.plot(time_tch[1], bf[1], lw=1.5, linestyle='-', alpha=0.6, color='k')
bf2 = np.ma.masked_greater(bf[2], .00081)
ax2.plot(time_tch[2][:8200], bf2[:8200], lw=1.5, linestyle='-', alpha=0.6, color='k')
ax2.plot(time_tch[3], bf[3], lw=1.5, linestyle='-', alpha=0.6, color='k', label=r'$N^2_{TS}$E')
ax2.plot(time_e, bf_temp_e, lw=2, linestyle='-', alpha=0.9, color='k', label=r'$N^2_T$E')
ax2.plot(time_w, bf_temp_w, lw=2, linestyle='-', alpha=0.8, color='darkorange', label=r'$N^2_T$W')
'''ax2.plot(time_1e, bf1_t[0], lw=2, linestyle='-', alpha=0.9, color='k', label=r'$N^2_T$E')
ax2.plot(time_2e, bf2_t[0], lw=2, linestyle='-', alpha=0.9, color='k')
ax2.plot(time_3e, bf3_t[0], lw=2, linestyle='-', alpha=0.9, color='k')
ax2.plot(time_4e, bf4_t[0], lw=2, linestyle='-', alpha=0.9, color='k')
ax2.plot(time_5e, bf5_t[0], lw=2, linestyle='-', alpha=0.9, color='k')
ax2.plot(time_1w, bf1_tw[0], lw=2, linestyle='-', alpha=0.8, color='darkorange')
ax2.plot(time_2w, bf2_tw[0], lw=2, linestyle='-', alpha=0.8, color='darkorange')
ax2.plot(time_3w, bf3_tw[0], lw=2, linestyle='-', alpha=0.8, color='darkorange')
ax2.plot(time_4w, bf4_tw[0], lw=2, linestyle='-', alpha=0.8, color='darkorange', label=r'$N^2_T$W')'''
ax2.legend(loc='upper center', frameon=True, bbox_to_anchor=(0.79, 0.99), ncol=3)
ax2.set_ylabel(r'[rad/s $10^{-4}$]')
ax2.set_ylim(0,0.0008)
ax2.set_yticks(np.arange(0, 0.0008+0.0002, 0.0002))
ax2.set_yticklabels(['0', '2','4','6'])
ax2.axvline(x = fi3, color='k', linestyle='-',lw=2)
ax2.axvline(x = fi4, color='k', linestyle='-',lw=2)
ax2.axvline(x = fi6, color='k', linestyle='-',lw=2)
ax2.text(0.02, 0.83, 'b)',
        transform=ax2.transAxes, bbox=dict(facecolor='white',
         alpha=0.9), color='k', fontsize=14)
fig.subplots_adjust(hspace=0.05)
ax.legend(loc='upper center', frameon=True, bbox_to_anchor=(0.79, 0.99), ncol=2)
ax.format_xdata = mpl.dates.DateFormatter('%b')
ax.xaxis.set_major_formatter(ax.format_xdata)
ax.text(735968, 33.4, centerify("NW")) #33.5
ax.text(735968-23, 32.6, centerify("monsoon")) #32.2
ax.text(736152, 33.4, centerify("SE"))
ax.text(736152-23, 32.6, centerify("monsoon"))
ax.text(736334, 33.4, centerify("NW"))
ax.text(736334-23, 32.6, centerify("monsoon"))
ax.text(736520, 33.4, centerify("SE"))
ax.text(736520-23, 32.6, centerify("monsoon"))
ax.text(736698, 33.4, centerify("NW"))
ax.text(736698-23, 32.6, centerify("monsoon"))
ax.text(736880, 33.4, centerify("SE"))
ax.text(736880-23, 32.6, centerify("monsoon"))
ax.text(737065, 33.4, centerify("NW"))
ax.text(737065-23, 32.6, centerify("monsoon"))
ax.set_rasterized(True)
ax = axs[2]
fi = datetime.datetime.strptime("01-03-2016", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-06-2016", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-09-2016", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-12-2016", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-03-2017", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-06-2017", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-09-2017", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-12-2017", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-03-2018", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-06-2018", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-09-2018", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-12-2018", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-03-2019", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.8, alpha=0.5)


levels = np.arange(27.7-3, 27.7+3 + 0.2, 0.2)
colori = cmocean.cm.thermal
ctop = ax.contourf(num2date(time_e), depth_new, temp_e.transpose(),
                 levels, cmap=colori, extend='both')
for c in ctop.collections:
    c.set_edgecolor("face")
    c.set_linewidth(0.000000000001)
ctop = ax.contourf(num2date(time_w_sh), depth_new, temp_w_sh.transpose(),
                 levels, cmap=colori, extend='both')
for c in ctop.collections:
    c.set_edgecolor("face")
    c.set_linewidth(0.000000000001)

cbar_ax = fig.add_axes([0.92, 0.07, 0.015, 0.33])
cbar = fig.colorbar(ctop, cax=cbar_ax, extend='both', ticks=[25, 27, 29, 31])
cbar.set_label(r'$^oC$')
cbar.solids.set_edgecolor("face")
cbar.outline.set_linewidth(0.000000000001)
ax.set_ylim(-24.5,-3)
ax.text(0.02, 0.83, 'c)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.9), color='k', fontsize=14)
ax.format_xdata = mpl.dates.DateFormatter('%b')
ax.xaxis.set_major_formatter(ax.format_xdata)
x1 = date2num(datetime.datetime.strptime('2016-03', '%Y-%m'))
x2 = date2num(datetime.datetime.strptime('2016-06', '%Y-%m'))
x3 = date2num(datetime.datetime.strptime('2016-09', '%Y-%m'))
x4 = date2num(datetime.datetime.strptime('2016-12', '%Y-%m'))
x5 = date2num(datetime.datetime.strptime('2017-03', '%Y-%m'))
x6 = date2num(datetime.datetime.strptime('2017-06', '%Y-%m'))
x7 = date2num(datetime.datetime.strptime('2017-09', '%Y-%m'))
x8 = date2num(datetime.datetime.strptime('2017-12', '%Y-%m'))
x9 = date2num(datetime.datetime.strptime('2018-03', '%Y-%m'))
x10 = date2num(datetime.datetime.strptime('2018-06', '%Y-%m'))
x11 = date2num(datetime.datetime.strptime('2018-09', '%Y-%m'))
x12 = date2num(datetime.datetime.strptime('2018-12', '%Y-%m'))
x13 = date2num(datetime.datetime.strptime('2019-03', '%Y-%m'))
ax.text(736128.3107449524, -32, '2016', fontsize=16)
ax.text(736493.9222557439, -32, '2017', fontsize=16)
ax.text(736857.389417498, -32, '2018', fontsize=16)
ax.text(737071.8243211883, -32, '2019', fontsize=16)
ax.set_xticks([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13])
ax.set_ylabel('depth [m]')
ax.axvline(fi3, color='k',linestyle='solid', lw=2)
ax.axvline(fi4, color='k', linestyle='solid', lw=2)
ax.axvline(fi6, color='k', linestyle='solid', lw=2)
ax.plot(time_e[300], -4.96, ">", color='k', markersize=8)
ax.plot(time_e[300], -8.32, ">", color='k', markersize=8)
ax.plot(time_e[300], -11.68, ">", color='k',markersize=8)
ax.plot(time_e[300], -15.05, ">", color='k', markersize=8)
ax.plot(time_e[300], -18.4, ">", color='k', markersize=8)
ax.plot(time_e[300], -21.76, ">", color='k', markersize=8)
ax.plot(time_e[300], -24.16, ">", color='k', markersize=8)
fig.subplots_adjust(left=0.08, hspace=0.15)
ax.set_rasterized(True)
# plt.savefig('/Users/carolina/Dropbox/Seychelles_Paper_Alma/figure_10.pdf')

bf_w = bf_temp_e
time_bf = time_e
mat = np.load('../../ssh/ssh_sat_adcp_new.npz')
pres_1E = mat['pres_1E']
pres_2E = mat['pres_2E']
pres_3E = mat['pres_3E']
pres_N = mat['pres_N']
tsat = mat['tsat']
ssh = mat['ssh']
tg1_ep = mat['time_1e']
tg2_ep = mat['time_2e']
tg3_ep = mat['time_3e']
tg_np = mat['time_N']

# Correlation with SSH Satellite
time, I1, I2 = colloc_time(time_bf, tsat, 24)
r, n, r0, t = nanxcorrcoef(bf_w[I1],
                           ssh[I2], 95, 0)
print(t[r == np.min(r)])

# Correlation with SSH ADCPE
ssh = np.concatenate((pres_1E, pres_2E, pres_3E))
tsat = np.concatenate((tg1_ep, tg2_ep, tg3_ep))
time, I1, I2 = colloc_time(time_bf, tsat, 1/24)
r, n, r0, t = nanxcorrcoef(bf_w[I1],
                           ssh[I2], 95, 0)
print(t[r == np.min(r)])
