# band pass ADCP data
import sys
sys.path.append("../../codes/")
import numpy as np
from matplotlib import rcParams as rcp
import matplotlib.pyplot as plt
from matplotlib import cm
import netCDF4 as S
import matplotlib as mpl
# external codes needed
import cmocean
from slomo_processing import *

def centerify(text, width=-1):
    lines = text.split('\n')
    width = max(map(len, lines)) if width == -1 else width
    return '\n'.join(line.center(width) for line in lines)

rcp['axes.labelsize'] = 16.
rcp['xtick.labelsize'] = 16.
rcp['ytick.labelsize'] = 16.
rcp['lines.linewidth'] = 2.
rcp['xtick.top'] = True
rcp['xtick.labeltop'] = False
rcp['xtick.bottom'] = True
rcp['xtick.labelbottom'] = True

dt = 1 / 24
bpi =  168 # 168 is 7 days, 720 is 30 days

file = S.Dataset('/Users/carolina/SLOMO/clean_data/adcp_E_slomo_hourly_20162019.nc', mode='r')
u_e = file.variables['u'][:]
v_e = file.variables['v'][:]
tg1_e = file.variables['time_vel'][:]
depth_e = file.variables['depth'][:]
ul1_e, vl1_e = make_band_pass_adcp(u_e, v_e, tg1_e, bpi, 'None', 'low')
ul1_e[u_e.mask] = np.ma.masked
vl1_e[v_e.mask] = np.ma.masked
#tg1_e[u_e[:,0].mask] = np.ma.masked

# load ADCP N
file = S.Dataset('/Users/carolina/SLOMO/clean_data/adcp_N_slomo_hourly_20162019.nc', mode='r')
u_n = file.variables['u'][:]
v_n = file.variables['v'][:]
tg_n1 = file.variables['time'][:]
ul1_n, vl1_n = make_band_pass_adcp(u_n, v_n, tg_n1, bpi, 'None', 'low')
ul1_n[u_n.mask] = np.ma.masked
vl1_n[v_n.mask] = np.ma.masked

# calculate du/dz, dv/dz
du_dz = np.diff(ul1_e, axis=1) / np.median(np.diff(depth_e))
dv_dz = np.diff(vl1_e, axis=1) / np.median(np.diff(depth_e))
s2_1 = du_dz ** 2 + dv_dz ** 2

ag = 0.8
fi1 = date2num(datetime.datetime.strptime('2016-01', '%Y-%m'))
fi2 = date2num(datetime.datetime.strptime('2018-07', '%Y-%m'))
fi3 = date2num(datetime.datetime.strptime('2017-01', '%Y-%m'))
fi4 = date2num(datetime.datetime.strptime('2018-01', '%Y-%m'))
fi6 = date2num(datetime.datetime.strptime('2019-01', '%Y-%m'))
fi5 = date2num(datetime.datetime.strptime('2019-06', '%Y-%m'))


levels = np.arange(-50,55,5)
ag = 0.3
fig, axs = plt.subplots(5, 1, sharex=True,figsize=(13.9,9))
ax = axs[0]
fi = datetime.datetime.strptime("01-03-2016", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-06-2016", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-09-2016", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-12-2016", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-03-2017", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-06-2017", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-09-2017", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-12-2017", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-03-2018", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-06-2018", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-09-2018", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-12-2018", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-03-2019", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
ax.axhline(y=0, color='k', linestyle=':', lw=0.5, alpha=0.8)
ax.text(735968, 67, centerify("NW")) # 86
ax.text(735968-23, 55.3, centerify("monsoon")) #77
ax.text(736152, 67, centerify("SE"))
ax.text(736152-23, 55.3, centerify("monsoon"))
ax.text(736334, 67, centerify("NW"))
ax.text(736334-23, 55.3, centerify("monsoon"))
ax.text(736520, 67, centerify("SE"))
ax.text(736520-23, 55.3, centerify("monsoon"))
ax.text(736698, 67, centerify("NW"))
ax.text(736698-23, 55.3, centerify("monsoon"))
ax.text(736880, 67, centerify("SE"))
ax.text(736880-23, 55.3, centerify("monsoon"))
ax.text(737065, 67, centerify("NW"))
ax.text(737065-23, 55.3, centerify("monsoon"))
ax.axvline(fi3, color='k',linestyle='solid', lw=2, alpha=0.8)
ax.axvline(fi4, color='k', linestyle='solid', lw=2, alpha=0.8)
ax.axvline(fi6, color='k', linestyle='solid', lw=2, alpha=0.8)
ax.text(0.02, 0.88, 'a)Depth-averaged zonal [u] velocities [cm'+ r's$^{-1}$' +']',
        transform=ax.transAxes, fontsize=12)

ax.plot(num2date(tg1_e), u_e.mean(-1) * 100, color='k', alpha=ag, lw=0.8, label='ADCPE hourly') # at the surface 12 m
ax.plot(num2date(tg1_e), ul1_e.mean(-1) * 100, color='k', lw=2, label='ADCPE low-passed') # at the surface 12 m
ax.plot(num2date(tg_n1), ul1_n.mean(-1) * 100, color='magenta', lw=2, label='ADCPN low-passed') # at the surface 12 m
ax.set_ylabel('[cm/s]')
ax.set_xlim(fi1, tg1_e[-1])
ax.set_ylim(-50, 50)
ax.set_yticks([-25,0,25])
ax.set_yticks([-25,0,25])
ax.legend(loc='lower center', frameon=True, bbox_to_anchor=(0.50, -0.01), ncol=3)
ax.set_rasterized(True)
ax = axs[1]
fi = datetime.datetime.strptime("01-03-2016", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-06-2016", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-09-2016", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-12-2016", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-03-2017", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-06-2017", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-09-2017", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-12-2017", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-03-2018", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-06-2018", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-09-2018", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-12-2018", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-03-2019", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
ax.axhline(y=0, color='k', linestyle='-', lw=0.5, alpha=0.8)
ax.axvline(fi3, color='k',linestyle='solid', lw=2, alpha=0.8)
ax.axvline(fi4, color='k', linestyle='solid', lw=2, alpha=0.8)
ax.axvline(fi6, color='k', linestyle='solid', lw=2, alpha=0.8)
ax.text(0.02, 1.05, 'b)Depth-averaged meridional [v] velocities [cm'+ r's$^{-1}$' +']',
        transform=ax.transAxes, fontsize=12)
ax.axvline(fi6, color='k', linestyle='-', lw=2)

ax.plot(num2date(tg1_e), v_e.mean(-1) * 100, color='k', alpha=ag, lw=0.8) # at the surface 12 m
ax.plot(num2date(tg1_e), vl1_e.mean(-1) * 100, color='k', lw=2) # at the surface 12 m
ax.plot(num2date(tg_n1), vl1_n.mean(-1) * 100, color='magenta', lw=2) # at the surface 12 m
ax.set_ylim(-50, 50)
ax.set_yticks([-25,0,25])
ax.set_ylabel('[cm/s]')
ax.set_rasterized(True)
ax = axs[2]
fi = datetime.datetime.strptime("01-03-2016", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-06-2016", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-09-2016", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-12-2016", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-03-2017", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-06-2017", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-09-2017", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-12-2017", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-03-2018", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-06-2018", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-09-2018", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-12-2018", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-03-2019", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
ax.text(0.02, 1.05, 'c)ADCPE zonal velocities [cm'+ r's$^{-1}$' +']',
        transform=ax.transAxes, fontsize=12)
ax.axvline(fi6, color='k', linestyle='-', lw=2)
ax.axvline(fi3, color='k',linestyle='solid', lw=2, alpha=0.8)
ax.axvline(fi4, color='k', linestyle='solid', lw=2, alpha=0.8)
ax.axvline(fi6, color='k', linestyle='solid', lw=2, alpha=0.8)

ctop = ax.contourf(num2date(tg1_e), depth_e, ul1_e.transpose() * 100,
                  levels, cmap=cmocean.cm.balance, extend='both')
ax.set_ylabel( 'depth [m]')
ax.format_xdata = mpl.dates.DateFormatter('%b-%y')
ax.xaxis.set_major_formatter(ax.format_xdata)
ax.set_xticklabels([])
ax.set_ylim(depth_e[0], depth_e[-1])
ax.set_rasterized(True)
ax = axs[3]
ctop = ax.contourf(num2date(tg1_e), depth_e, vl1_e.transpose() * 100,
                  levels, cmap=cmocean.cm.balance, extend='both')
ax.set_ylabel( 'depth [m]')
fig.subplots_adjust(right=0.90)
cbar_ax = fig.add_axes([0.92, 0.28, 0.015, 0.28])
cbar = fig.colorbar(ctop, cax=cbar_ax, extend='both', ticks=[-40, -20, 0, 20, 40])
cbar.set_label('[cm/s]',labelpad=-1)
ax.axvline(fi3, color='k',linestyle='solid', lw=2, alpha=0.8)
ax.axvline(fi4, color='k', linestyle='solid', lw=2, alpha=0.8)
ax.axvline(fi6, color='k', linestyle='solid', lw=2, alpha=0.8)
ax.format_xdata = mpl.dates.DateFormatter('%b-%y')
ax.xaxis.set_major_formatter(ax.format_xdata)
ax.set_xticklabels([])
ax.text(0.02, 1.05, 'd)ADCPE meridional velocities [cm'+ r's$^{-1}$' +']',
        transform=ax.transAxes, fontsize=12)
ax.format_xdata = mpl.dates.DateFormatter('%b')
ax.xaxis.set_major_formatter(ax.format_xdata)
ax.set_ylim(depth_e[0], depth_e[-1])

ax = axs[4]
levels = np.arange(0, 0.6e-3 + 0.05e-3, 0.05e-3)
ctop = ax.contourf(num2date(tg1_e), depth_e[1:-1], s2_1.transpose()[:-1],
                  levels, cmap=cm.afmhot_r, extend='max')
for c in ctop.collections:
    c.set_edgecolor("face")
    c.set_linewidth(0.000000000001)
ax.set_rasterized(True)
ax.set_ylabel( 'depth [m]')
cbar_ax = fig.add_axes([0.92, 0.08, 0.015, 0.18])
cbar = fig.colorbar(ctop, cax=cbar_ax, extend='both', ticks=levels[::4][:-1])
cbar.set_ticklabels([0,2,4])
ax.text(1.045, 1.05, r'10$^{-4}$', transform=ax.transAxes, fontsize=10)
cbar.set_label(r'[$s^{-2}$]',labelpad=10)
ax.set_ylabel('depth [m]')
ax.text(0.02, 1.05, 'e)ADCPE S$^2$ ['+ r's$^{-2}$' +']',
        transform=ax.transAxes, fontsize=12)
fi = datetime.datetime.strptime("01-03-2016", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-06-2016", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-09-2016", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-12-2016", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-03-2017", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-06-2017", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-09-2017", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-12-2017", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-03-2018", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-06-2018", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-09-2018", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-12-2018", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
fi = datetime.datetime.strptime("01-03-2019", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw = 0.8, alpha=0.5)
ax.axvline(fi3, color='k',linestyle='solid', lw=2, alpha=0.8)
ax.axvline(fi4, color='k', linestyle='solid', lw=2, alpha=0.8)
ax.axvline(fi6, color='k', linestyle='solid', lw=2, alpha=0.8)
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
ax.text(736128.3107449524, -34, '2016', fontsize=16)
ax.text(736493.9222557439, -34, '2017', fontsize=16)
ax.text(736857.389417498, -34, '2018', fontsize=16)
ax.text(737071.8243211883, -34, '2019', fontsize=16)
ax.set_xticks([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13])
fig.subplots_adjust(left=0.08)
# plt.savefig('/Users/carolina/Dropbox/Seychelles_Paper_Alma/fig_timeseries_lf_large_test_wshear.pdf')
