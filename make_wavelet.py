# wavelet for ADCP data SLOMO
# monte carlo for wavelet analysis
import sys
sys.path.append("../../codes/")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams as rcp
import matplotlib as mpl
from matplotlib.dates import date2num, num2date, datestr2num, datetime
from matplotlib import cm
import netCDF4 as S
# external codes needed
import pycwt as wavelet
import cmocean
from slomo_processing import *

rcp['font.family'] = 'sans serif'
rcp['axes.labelsize'] = 16.
rcp['xtick.labelsize'] = 14.
rcp['ytick.labelsize'] = 14.

def centerify(text, width=-1):
    lines = text.split('\n')
    width = max(map(len, lines)) if width == -1 else width
    return '\n'.join(line.center(width) for line in lines)

savetitle = 'wavelet_e_cw.pdf'
b1 = 4 * 24 # 117
b2 = 7 * 24 # 1260
t10 = 736297.1453356482
t20 = 736471.173599537

lat = -4.669
lon =  55.646
file = S.Dataset('/Users/carolina/SLOMO/clean_data/adcp_E_slomo_hourly_20162019.nc', mode='r')
u = file.variables['u'][:]
v = file.variables['v'][:]
time = file.variables['time_vel'][:]
depth_e = file.variables['depth'][:]
f, b = coriolis(lat)
fday = f * 86400 / (2 * np.pi)

alpha = 0.0                          # Lag-1 autocorrelation for white noise
dj = 1 / 12                          # Twelve sub-octaves per octaves 1/12 is the correct
dt = time[1] - time[0]
s0 = 4 * dt                       	 # Starting scale 4 * dt for hourly data
J = -1                       # end scale is around 10 days if 7/dj, 6 days if 6/dj
mother = wavelet.Morlet(6)           # Morlet mother wavelet with m=6, smaller m thinner energy in time, more power
avg1, avg2 = (b1 / 24, b2 / 24)      # average period in days
slevel = 0.95

umean = u.mean(-1) * 100#ug1[:, 3] * 100
vmean = v.mean(-1) * 100
varu = masked_interp_single(time, umean)  # masked arrays are interpolated
varv = masked_interp_single(time, vmean)  # masked arrays are interpolated
#var = np.conjugate(umean + vmean * 1j)  # conjugate is cw, normal is for ccw
var = umean + vmean * 1j
var = varu
tit = 'u'
# var = np.sqrt(varu**2 + varv**2)
N = var.size                         # Number of measurements
#dj will change the time/freq resolution
std = var.std()                      # Standard deviation
std2 = std ** 2                      # Variance
var = (var - var.mean()) / std       # Calculating anomaly and normalizing

wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(var, dt, dj, s0, J,
                                                      mother)
period = 1 / freqs
powerf = (np.abs(wave)) ** 2
global_ws = (np.sum(powerf, axis=1) / N)  # time-average over all times

# Global wavelet spectrum & significance levels:
dof = N - scales  # the -scale corrects for padding at edges
global_signif = wavelet.significance(var, dt, scales, 0, alpha,
    significance_level=slevel, wavelet=mother)

signif, fft_theor = wavelet.significance(var, dt, scales, 0, alpha,
                        significance_level=slevel, wavelet=mother)
sig95 = np.ones([1, N]) * signif[:, None]
sig95 = powerf / sig95

fi1 = 1 / (b1 / 24)
fi2 = 1 / (b2 / 24)
fi = np.where((freqs > fi2) & (freqs < fi1))
print('The freq resolution between ' + str(b1) + ' and ' + str(b2) + ' is ', np.abs(np.diff(freqs[fi]).mean()),'cpd')
print('The freq resolution is ', np.abs(np.diff(freqs).mean()),' cpd')
fi1 = b1 / 24
fi2 = b2 / 24
fi = np.where((period > fi1) & (period < fi2))
print('The period resolution between ' + str(b1) + ' and ' + str(b2) + ' is ', np.abs(np.diff(period[fi]).mean())*24,' hours')
print('The period resolution is', np.abs(np.diff(period).min())*24,' hours')
sel = mpl.mlab.find((period >= avg1) & (period < avg2))
scale_avg = (scales * np.ones((N, 1))).transpose()
# As in Torrence and Compo (1998) equation 24
scale_avg = powerf / scale_avg
Cdelta = mother.cdelta
scale_avg = std2 * dj * dt / Cdelta * scale_avg[sel, :].sum(axis=0)
scale_avg_signif, tmp = wavelet.significance(std2, dt, scales, 2, alpha,
                            significance_level=slevel, dof=[scales[sel[0]],
                            scales[sel[-1]]], wavelet=mother)
scale_avg_u = scale_avg
tg1u = time

ticks_x = [date2num(datetime.datetime.strptime("01-2016", "%m-%Y")),
date2num(datetime.datetime.strptime("04-2016", "%m-%Y")),
date2num(datetime.datetime.strptime("07-2016", "%m-%Y")),
date2num(datetime.datetime.strptime("10-2016", "%m-%Y")),
date2num(datetime.datetime.strptime("01-2017", "%m-%Y")),
date2num(datetime.datetime.strptime("04-2017", "%m-%Y")),
date2num(datetime.datetime.strptime("07-2017", "%m-%Y")),
date2num(datetime.datetime.strptime("10-2017", "%m-%Y")),
date2num(datetime.datetime.strptime("01-2018", "%m-%Y")),
date2num(datetime.datetime.strptime("04-2018", "%m-%Y"))
]

fi = datetime.datetime.strptime("01-01-2017", "%d-%m-%Y")
fi2 = datetime.datetime.strptime("01-01-2018", "%d-%m-%Y")
fi3 = datetime.datetime.strptime("01-01-2019", "%d-%m-%Y")
levels = np.logspace(0,3.1,20)
fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(11, 8.26))
ax1 = axs[0]
ax1.text(0.03, 0.88, 'a',
        transform=ax1.transAxes, bbox=dict(facecolor='white',
         alpha=0.9), color='k', fontsize=14)
ax1.text(0.03, 0.05, 'u',
        transform=ax1.transAxes, color='k', fontsize=14)
ax1.axvspan(t10, t20, alpha=0.1, color='black')
ctop = ax1.contourf(time, period, np.log2(powerf),
	                np.log2(levels), cmap=cmocean.cm.speed, extend='max')
ax1.contour(time, period, np.log2(powerf),
	                np.arange(5,11,1), linewidths=0.5, colors='k')
ax1.axhline(y=1 / fday * -1, color='k', linestyle='dashed', linewidth=0.5)
ax1.axvline(x=date2num(fi), color='k', linestyle='-',lw=1, alpha=0.8)
ax1.axvline(x=date2num(fi2), color='k', linestyle='-', lw=1, alpha=0.8)
ax1.axvline(x=date2num(fi3), color='k', linestyle='-', lw=1, alpha=0.8)
ax1.plot(time, coi, linewidth=0.8, color='k')
ax1.fill_between(time, coi, 160, color='grey', alpha='0.8')
ax1.format_xdata = mpl.dates.DateFormatter('%b-%y')
ax1.xaxis.set_major_formatter(ax1.format_xdata)
ax1.set_ylabel('days')
ax1.set_yscale('log')
ax1.set_ylim([0.3, 160])
ax1.set_yticks([0.5, 1, 3, 6.1, 10, 15, 30, 60, 100])
ax1.set_xticks(ticks_x)
ax1.set_yticklabels((r'$M_2$', r'$K_1$', '3',r'$f$','10', '15','30', '60', '100'))
ax1.set_rasterized(True)
ax1.text(735968, 260, centerify("NW")) #320
ax1.text(735968-40, 180, centerify("monsoon")) #360
ax1.text(736148, 260, centerify("SE"))
ax1.text(736152-40, 180, centerify("monsoon"))
ax1.text(736330, 260, centerify("NW"))
ax1.text(736334-40, 180, centerify("monsoon"))
ax1.text(736516, 260, centerify("SE"))
ax1.text(736520-40, 180, centerify("monsoon"))
ax1.text(736694, 260, centerify("NW"))
ax1.text(736698-40, 180, centerify("monsoon"))
ax1.text(736874, 260, centerify("SE"))
ax1.text(736880-40, 180, centerify("monsoon"))
ax1.text(737061, 260, centerify("NW"))
ax1.text(737065-40, 180, centerify("monsoon"))
fi = datetime.datetime.strptime("01-03-2016", "%d-%m-%Y")
ax1.axvline(x=date2num(fi), color='k', linestyle='--', lw=0.8, alpha=0.3)
fi = datetime.datetime.strptime("01-06-2016", "%d-%m-%Y")
ax1.axvline(x=date2num(fi), color='k', linestyle='--', lw=0.8, alpha=0.3)
fi = datetime.datetime.strptime("01-09-2016", "%d-%m-%Y")
ax1.axvline(x=date2num(fi), color='k', linestyle='--', lw=0.8, alpha=0.3)
fi = datetime.datetime.strptime("01-12-2016", "%d-%m-%Y")
ax1.axvline(x=date2num(fi), color='k', linestyle='--', lw=0.8, alpha=0.3)
fi = datetime.datetime.strptime("01-03-2017", "%d-%m-%Y")
ax1.axvline(x=date2num(fi), color='k', linestyle='--', lw=0.8, alpha=0.3)
fi = datetime.datetime.strptime("01-06-2017", "%d-%m-%Y")
ax1.axvline(x=date2num(fi), color='k', linestyle='--', lw=0.8, alpha=0.3)
fi = datetime.datetime.strptime("01-09-2017", "%d-%m-%Y")
ax1.axvline(x=date2num(fi), color='k', linestyle='--', lw=0.8, alpha=0.3)
fi = datetime.datetime.strptime("01-12-2017", "%d-%m-%Y")
ax1.axvline(x=date2num(fi), color='k', linestyle='--', lw=0.8, alpha=0.3)
fi = datetime.datetime.strptime("01-03-2018", "%d-%m-%Y")
ax1.axvline(x=date2num(fi), color='k', linestyle='--', lw=0.8, alpha=0.3)
fi = datetime.datetime.strptime("01-06-2018", "%d-%m-%Y")
ax1.axvline(x=date2num(fi), color='k', linestyle='--', lw=0.8, alpha=0.3)
fi = datetime.datetime.strptime("01-09-2018", "%d-%m-%Y")
ax1.axvline(x=date2num(fi), color='k', linestyle='--', lw=0.8, alpha=0.3)
fi = datetime.datetime.strptime("01-12-2018", "%d-%m-%Y")
ax1.axvline(x=date2num(fi), color='k', linestyle='--', lw=0.8, alpha=0.3)
fi = datetime.datetime.strptime("01-03-2019", "%d-%m-%Y")
ax1.axvline(x=date2num(fi), color='k', linestyle='--', lw=0.8, alpha=0.3)
ax1.plot(time[-1]-2, 0.1666, '<', color='k', ms=14)
ax1.plot(time[-1]-2, 2, '<', color='k', ms=14)
ax1.plot(time[-1]-2, 7, '<', color='k', ms=14)

tit = 'v'
var = varv
# var = np.sqrt(varu**2 + varv**2)
N = var.size                         # Number of measurements
# dj will change the time/freq resolution
std = var.std()                      # Standard deviation
std2 = std ** 2                      # Variance
var = (var - var.mean()) / std       # Calculating anomaly and normalizing
wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(var, dt, dj, s0, J,
                                                      mother)
period = 1 / freqs
powerf = (np.abs(wave)) ** 2
global_ws = (np.sum(powerf, axis=1) / N)  # time-average over all times

# Global wavelet spectrum & significance levels:
dof = N - scales  # the -scale corrects for padding at edges
global_signif = wavelet.significance(var, dt, scales, 0, alpha,
    significance_level=slevel, wavelet=mother)

signif, fft_theor = wavelet.significance(var, dt, scales, 0, alpha,
                        significance_level=slevel, wavelet=mother)
sig95 = np.ones([1, N]) * signif[:, None]
sig95 = powerf / sig95

fi1 = 1 / (b1 / 24)
fi2 = 1 / (b2 / 24)
fi = np.where((freqs > fi2) & (freqs < fi1))
print('The freq resolution between ' + str(b1) + ' and ' + str(b2) + ' is ', np.abs(np.diff(freqs[fi]).mean()),'cpd')
print('The freq resolution is ', np.abs(np.diff(freqs).mean()),' cpd')
fi1 = b1 / 24
fi2 = b2 / 24
fi = np.where((period > fi1) & (period < fi2))

print('The period resolution between ' + str(b1) + ' and ' + str(b2) + ' is ', np.abs(np.diff(period[fi]).mean())*24,' hours')
print('The period resolution is', np.abs(np.diff(period).min())*24,' hours')
sel = mpl.mlab.find((period >= avg1) & (period < avg2))
scale_avg = (scales * np.ones((N, 1))).transpose()
# As in Torrence and Compo (1998) equation 24
scale_avg = powerf / scale_avg
Cdelta = mother.cdelta
scale_avg = std2 * dj * dt / Cdelta * scale_avg[sel, :].sum(axis=0)
scale_avg_signif, tmp = wavelet.significance(std2, dt, scales, 2, alpha,
                            significance_level=slevel, dof=[scales[sel[0]],
                            scales[sel[-1]]], wavelet=mother)

fi = datetime.datetime.strptime("01-01-2017", "%d-%m-%Y")
fi2 = datetime.datetime.strptime("01-01-2018", "%d-%m-%Y")
fi3 = datetime.datetime.strptime("01-01-2019", "%d-%m-%Y")
ax1 = axs[1]
ax1.axvspan(t10, t20, alpha=0.2, color='black')
ctop = ax1.contourf(time, period, np.log2(powerf),
	                np.log2(levels), cmap=cmocean.cm.speed, extend='max')
ax1.contour(time, period, np.log2(powerf),
	                np.arange(5,11,1), linewidths=0.5, colors='k')
ax1.axhline(y=1 / fday * -1, color='k', linestyle='dashed', linewidth=0.5)
ax1.axvline(x=date2num(fi), color='k', linestyle='-',lw=1, alpha=0.8)
ax1.axvline(x=date2num(fi2), color='k', linestyle='-', lw=1, alpha=0.8)
ax1.axvline(x=date2num(fi3), color='k', linestyle='-', lw=1, alpha=0.8)
ax1.plot(time, coi, linewidth=0.8, color='k')
ax1.fill_between(time, coi, 160, color='grey', alpha='0.8')
ax1.format_xdata = mpl.dates.DateFormatter('%b')
ax1.xaxis.set_major_formatter(ax1.format_xdata)
ax1.set_ylabel('days')
ax1.set_yscale('log')
fig.subplots_adjust(left=0.07, right=0.9, hspace=0.08, wspace=0.18)
cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
cbar = fig.colorbar(ctop, cax=cbar_ax, extend='neither', ticks=[0,3,6,9])
cbar.set_ticklabels([0,'$2^3$','$2^6$','$2^9$'])
cbar.set_label(r'$[cm^2 s^{-2}]$')
ax1.set_ylim([0.1, 160])
ax1.set_yticks([0.5, 1, 3, 6.1, 10, 15, 30, 60, 100])
ax1.set_xticks(ticks_x)
ax1.set_yticklabels((r'$M_2$', r'$K_1$', '3',r'$f$','10', '15','30', '60', '100'))
ax1.set_rasterized(True)
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
ax1.text(736128.3107449524, 0.03, '2016', fontsize=16)
ax1.text(736493.9222557439, 0.03, '2017', fontsize=16)
ax1.text(736857.389417498, 0.03, '2018', fontsize=16)
ax1.text(737071.8243211883, 0.03, '2019', fontsize=16)
ax1.set_xticks([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13])
ax1.text(0.03, 0.88, 'b',
        transform=ax1.transAxes, bbox=dict(facecolor='white',
         alpha=0.9), color='k', fontsize=14)
ax1.text(0.03, 0.05, 'v',
        transform=ax1.transAxes, color='k', fontsize=14)
ax1.plot(time[-1]-2, 0.1666, '<', color='k', ms=14)
ax1.plot(time[-1]-2, 2, '<', color='k', ms=14)
ax1.plot(time[-1]-2, 7, '<', color='k', ms=14)

fi = datetime.datetime.strptime("01-03-2016", "%d-%m-%Y")
ax1.axvline(x=date2num(fi), color='k', linestyle='--', lw=0.8, alpha=0.3)
fi = datetime.datetime.strptime("01-06-2016", "%d-%m-%Y")
ax1.axvline(x=date2num(fi), color='k', linestyle='--', lw=0.8, alpha=0.3)
fi = datetime.datetime.strptime("01-09-2016", "%d-%m-%Y")
ax1.axvline(x=date2num(fi), color='k', linestyle='--', lw=0.8, alpha=0.3)
fi = datetime.datetime.strptime("01-12-2016", "%d-%m-%Y")
ax1.axvline(x=date2num(fi), color='k', linestyle='--', lw=0.8, alpha=0.3)
fi = datetime.datetime.strptime("01-03-2017", "%d-%m-%Y")
ax1.axvline(x=date2num(fi), color='k', linestyle='--', lw=0.8, alpha=0.3)
fi = datetime.datetime.strptime("01-06-2017", "%d-%m-%Y")
ax1.axvline(x=date2num(fi), color='k', linestyle='--', lw=0.8, alpha=0.3)
fi = datetime.datetime.strptime("01-09-2017", "%d-%m-%Y")
ax1.axvline(x=date2num(fi), color='k', linestyle='--', lw=0.8, alpha=0.3)
fi = datetime.datetime.strptime("01-12-2017", "%d-%m-%Y")
ax1.axvline(x=date2num(fi), color='k', linestyle='--', lw=0.8, alpha=0.3)
fi = datetime.datetime.strptime("01-03-2018", "%d-%m-%Y")
ax1.axvline(x=date2num(fi), color='k', linestyle='--', lw=0.8, alpha=0.3)
fi = datetime.datetime.strptime("01-06-2018", "%d-%m-%Y")
ax1.axvline(x=date2num(fi), color='k', linestyle='--', lw=0.8, alpha=0.3)
fi = datetime.datetime.strptime("01-09-2018", "%d-%m-%Y")
ax1.axvline(x=date2num(fi), color='k', linestyle='--', lw=0.8, alpha=0.3)
fi = datetime.datetime.strptime("01-12-2018", "%d-%m-%Y")
ax1.axvline(x=date2num(fi), color='k', linestyle='--', lw=0.8, alpha=0.3)
fi = datetime.datetime.strptime("01-03-2019", "%d-%m-%Y")
ax1.axvline(x=date2num(fi), color='k', linestyle='--', lw=0.8, alpha=0.3)

# plt.savefig(savetitle, bbox_inches='tight')
