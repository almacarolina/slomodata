# band pass ADCP data
import sys
sys.path.append("../../codes")
import numpy as np
import matplotlib as mpl
from matplotlib import rcParams as rcp
import matplotlib.pyplot as plt
from matplotlib import cm as colorm
import scipy.io as sci
import netCDF4 as S
from matplotlib.dates import date2num, num2date, datestr2num, datetime
# external codes needed
import pycwt as wavelet
import cmocean
from slomo_processing import *

# rcp['font.size'] = 16
rcp['axes.labelsize'] = 16.
rcp['xtick.labelsize'] = 16.
rcp['ytick.labelsize'] = 16.
rcp['lines.linewidth'] = 1.
rcp['xtick.top'] = True
rcp['xtick.labeltop'] = False
rcp['xtick.bottom'] = True
rcp['xtick.labelbottom'] = True
rcp['font.family'] = 'sans serif'


def centerify(text, width=-1):
    lines = text.split('\n')
    width = max(map(len, lines)) if width == -1 else width
    return '\n'.join(line.center(width) for line in lines)

# load satellite data from program make_fig_ssh_ADCPsat.py in SLOMO/ssh/
#README='ssh from SAT atop Plateau 55E -4.5S and ADCPE and ADCPN low passed 7 days'
mat = np.load('../../ssh/ssh_sat_adcp_new.npz')
tsat = mat['tsat']
ssh = mat['ssh']

# load satellite climatology near ADCP location from
# ./codes/model_comparison_avisopop_large.py
mat = np.load('../../codes/ssh_clim_plateau_1993_2009.npz')
ssh_clim = mat['ssh_clim']
tstart = date2num(datetime.datetime.strptime('2016-01-15', '%Y-%m-%d'))
tend = date2num(datetime.datetime.strptime('2017-01-15', '%Y-%m-%d'))
time_clim = np.arange(tstart, tstart+30*12, 30)
tstart = date2num(datetime.datetime.strptime('2017-01-15', '%Y-%m-%d'))
tend = date2num(datetime.datetime.strptime('2018-01-15', '%Y-%m-%d'))
time_clim2 = np.arange(tstart, tstart+30*12, 30)
tstart = date2num(datetime.datetime.strptime('2018-01-15', '%Y-%m-%d'))
tend = date2num(datetime.datetime.strptime('2019-01-15', '%Y-%m-%d'))
time_clim3 = np.arange(tstart, tstart+30*12, 30)
tstart = date2num(datetime.datetime.strptime('2019-01-15', '%Y-%m-%d'))
tend = date2num(datetime.datetime.strptime('2020-01-15', '%Y-%m-%d'))
time_clim4 = np.arange(tstart, tstart+30*12, 30)
t_clim = np.concatenate([time_clim, time_clim2, time_clim3, time_clim4])
ssh_clim = np.concatenate([ssh_clim, ssh_clim, ssh_clim, ssh_clim])

# load model climatology
# SLOMO/POP/ssh_ek_all.py
mat = np.load('../../POP/ssh_clim_pop_19932009_point.npz')
ssh_clim_pop = mat['ssh_clim']
ssh_clim_pop  = ssh_clim_pop - ssh_clim_pop.mean()
ssh_clim_pop = np.concatenate([ssh_clim_pop, ssh_clim_pop, ssh_clim_pop, ssh_clim_pop])


# load wind
file = np.loadtxt(fname='../../winds/wind_mahe_2018_2019.csv', delimiter=',')
year = file[:, 0]
month = file[:, 1]
day = file[:,2]
hour = file[:,3]
dir = file[:, 4]
speed = file[:,5] / 1.94384 # Convert m/s to knots
u_wind1 = speed * np.cos(np.deg2rad(dir))
v_wind1 = speed * np.sin(np.deg2rad(dir))
time1 = np.zeros(year.shape[0])
for ii in range(year.shape[0]):
    time1[ii] = date2num(datetime.datetime(np.int(year[ii]), np.int(month[ii]),
                         np.int(day[ii]), np.int(hour[ii])))
mat = sci.loadmat('../../winds/winds.mat')
wind = mat['wind_MO']
u_wind = wind['u'][0,0]
v_wind = wind['v'][0,0]
time = wind['time'][0,0] - 366 # we need this to change to python time
fi1 = date2num(datetime.datetime.strptime('2016-01', '%Y-%m'))
fi2 = date2num(datetime.datetime.strptime('2018-06-30', '%Y-%m-%d'))
idxa = np.where(time > fi1)[0][0]
idxb = np.where(time > fi2)[0][0]
ui = u_wind[idxa:398789+1].squeeze()
vi = v_wind[idxa:398789+1].squeeze()
ti = time[idxa:398789+1].squeeze()
ui = np.concatenate((ui, u_wind1))
vi = np.concatenate((vi, v_wind1))
ti = np.concatenate((ti, time1))

# load pressure
file = S.Dataset('/Users/carolina/SLOMO/clean_data/adcp_E_slomo_hourly_20162019.nc', mode='r')
press_e = np.ma.masked_equal(file.variables['pressure'][:], 0)
time_e = file.variables['time_press'][:]
depth_e = file.variables['depth'][:]

lin = 20
ag = 0.8
fi1 = date2num(datetime.datetime.strptime('2016-01', '%Y-%m'))
fi2 = date2num(datetime.datetime.strptime('2019-03', '%Y-%m'))
fi3 = date2num(datetime.datetime.strptime('2017-01', '%Y-%m'))
fi4 = date2num(datetime.datetime.strptime('2018-01', '%Y-%m'))
fi5 = date2num(datetime.datetime.strptime('2019-01', '%Y-%m'))
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(13.9,4.64))
ax = axs[0]
q = ax.quiver(num2date(ti[::lin]), 0, ui[::lin],
               vi[::lin], # 72 is for NDBC buoy, none for airport
               color='k',
               units='y',
               scale_units='y',
               scale = 1,
               headwidth = 8,
               headlength = 10,
               width = 0.1, label='wind')
ax.set_ylim(-10, 10)
ax.set_ylabel('[m/s]')
ax.legend(loc=3)
ax.text(0.02, 0.85, 'a)',
        transform=ax.transAxes, bbox=dict(facecolor='black',
         alpha=0.1), fontsize=14)

ax.axhline(y=0, color='k', linestyle='-', lw=0.5, alpha=0.8)
ax.axvline(x = fi3, color='k', linestyle='-',lw=1, alpha=0.8)
ax.axvline(x = fi4, color='k', linestyle='-',lw=1, alpha=0.8)
ax.axvline(x = fi5, color='k', linestyle='-',lw=1, alpha=0.8)
fi = datetime.datetime.strptime("01-03-2016", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.5, alpha=0.3)
fi = datetime.datetime.strptime("01-06-2016", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.5, alpha=0.3)
fi = datetime.datetime.strptime("01-09-2016", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.5, alpha=0.3)
fi = datetime.datetime.strptime("01-12-2016", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.5, alpha=0.3)
fi = datetime.datetime.strptime("01-03-2017", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.5, alpha=0.3)
fi = datetime.datetime.strptime("01-06-2017", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.5, alpha=0.3)
fi = datetime.datetime.strptime("01-09-2017", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.5, alpha=0.3)
fi = datetime.datetime.strptime("01-12-2017", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.5, alpha=0.3)
fi = datetime.datetime.strptime("01-03-2018", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.5, alpha=0.3)
fi = datetime.datetime.strptime("01-06-2018", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.5, alpha=0.3)
fi = datetime.datetime.strptime("01-09-2018", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.5, alpha=0.3)
fi = datetime.datetime.strptime("01-12-2018", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.5, alpha=0.3)
fi = datetime.datetime.strptime("01-03-2019", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.5, alpha=0.3)
ax.text(735968, 11.4, centerify("NW")) # 13.2
ax.text(735968-23, 10.3, centerify("monsoon")) # 12.1
ax.text(736152, 11.4, centerify("SE"))
ax.text(736152-23, 10.3, centerify("monsoon"))
ax.text(736334, 11.4, centerify("NW"))
ax.text(736334-23, 10.3, centerify("monsoon"))
ax.text(736520, 11.4, centerify("SE"))
ax.text(736520-23, 10.3, centerify("monsoon"))
ax.text(736698, 11.4, centerify("NW"))
ax.text(736698-23, 10.3, centerify("monsoon"))
ax.text(736880, 11.4, centerify("SE"))
ax.text(736880-23, 10.3, centerify("monsoon"))
ax.text(737065, 11.4, centerify("NW"))
ax.text(737065-23, 10.3, centerify("monsoon"))

ax = axs[1]
ax.axhline(y=0, color='k', linestyle='-', lw=0.5, alpha=0.8)
ax.plot(num2date(t_clim), ssh_clim * 100, 'k', alpha=0.5, label='AVISO clim', lw=2)
ax.plot(num2date(t_clim), ssh_clim_pop, 'k', linestyle='--', alpha=0.5, label='POP clim', lw=2)
ax.plot(num2date(tsat), ssh, color='red',lw=2, label='AVISO')
ax.plot(num2date(time_e), press_e, 'k', lw=2, label='ADCPE')

ax.set_ylim(-25, 30)
ax.set_ylabel('[cm]')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=4)
ax.axvline(x = fi3, color='k', linestyle='-',lw=1, alpha=0.8)
ax.axvline(x = fi4, color='k', linestyle='-',lw=1, alpha=0.8)
ax.axvline(x = fi5, color='k', linestyle='-',lw=1, alpha=0.8)
ax.text(0.02, 0.85, 'b)',
        transform=ax.transAxes, bbox=dict(facecolor='black',
         alpha=0.1), fontsize=14)
fig.subplots_adjust(hspace=0.05)
fi = datetime.datetime.strptime("01-03-2016", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.5, alpha=0.3)
fi = datetime.datetime.strptime("01-06-2016", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.5, alpha=0.3)
fi = datetime.datetime.strptime("01-09-2016", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.5, alpha=0.3)
fi = datetime.datetime.strptime("01-12-2016", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.5, alpha=0.3)
fi = datetime.datetime.strptime("01-03-2017", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.5, alpha=0.3)
fi = datetime.datetime.strptime("01-06-2017", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.5, alpha=0.3)
fi = datetime.datetime.strptime("01-09-2017", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.5, alpha=0.3)
fi = datetime.datetime.strptime("01-12-2017", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.5, alpha=0.3)
fi = datetime.datetime.strptime("01-03-2018", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.5, alpha=0.3)
fi = datetime.datetime.strptime("01-06-2018", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.5, alpha=0.3)
fi = datetime.datetime.strptime("01-09-2018", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.5, alpha=0.3)
fi = datetime.datetime.strptime("01-12-2018", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.5, alpha=0.3)
fi = datetime.datetime.strptime("01-03-2019", "%d-%m-%Y")
ax.axvline(x=date2num(fi), color='k', linestyle='dashed', lw=0.5, alpha=0.3)
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
ax.text(736128.3107449524, -38, '2016', fontsize=16)
ax.text(736493.9222557439, -38, '2017', fontsize=16)
ax.text(736857.389417498, -38, '2018', fontsize=16)
ax.text(737071.8243211883, -38, '2019', fontsize=16)
ax.set_xticks([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13])
ax.set_xlim(fi1, fi2)
# plt.savefig('/Users/carolina/Dropbox/Seychelles_Paper_Alma/figure_6.pdf', bbox_inches='tight')

mat = np.load('../../CORE/wind_model_obs.npz') # from read_CORE_data.py in SLOMO/CORE
u_obs = mat['u_obs']
v_obs = mat['v_obs']
u_mod = mat['u_model']
v_mod = mat['v_model']
t_obs = mat['t_obs']
t_mod = mat['t_model']

time_both, I1, I2 = colloc_time(t_obs, t_mod, 1/24)
time_both = time_both[:-1]
uob = u_obs[I1[:-1]]
vob = v_obs[I1[:-1]]
umod = u_mod[I2[:-1]]
vmod = v_mod[I2[:-1]]


udif = umod - uob.squeeze()
vdif = vmod - vob.squeeze()

lowi = 24 * 15 # 7 days

uob_low = make_band_pass_1d(uob.squeeze(), time_both, lowi, 'None', 'low')
vob_low = make_band_pass_1d(vob.squeeze(), time_both, lowi, 'None', 'low')
umod_low = make_band_pass_1d(umod[:], time_both, lowi, 'None', 'low')
vmod_low = make_band_pass_1d(vmod[:], time_both, lowi, 'None', 'low')


lin = 10
fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(13.9, 3.2))
ax = axs[0]
ax.plot(num2date(time_both), uob_low, label='Observation', lw=1)
ax.plot(num2date(time_both), umod_low, label='Model')
ax.legend(loc=1)
ax.set_ylim(-10, 10)
ax.set_title('u')
ax.set_ylabel('[m/s]')
ax.axhline(y=0, color='k', alpha=0.5, lw=0.5)

ax = axs[1]
ax.plot(num2date(time_both), vob_low)
ax.plot(num2date(time_both), vmod_low)
fi1 = date2num(datetime.datetime.strptime('1993-01-01', '%Y-%m-%d'))
fi2 = date2num(datetime.datetime.strptime('2009-12-31', '%Y-%m-%d'))
ax.set_xlim(fi1, fi2)
ax.set_ylim(-10, 10)
ax.set_title('v')
fig.suptitle('Wind comparison')
ax.set_ylabel('[m/s]')
ax.axhline(y=0, color='k', alpha=0.5, lw=0.5)
plt.savefig('fig_timeseries_model_obs_comparison.pdf', bbox_inches='tight')


lin = 10
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(13.9, 3))
ax = axs[0]
q = ax.quiver(num2date(time_both[::lin]), 0, uob[::lin,0],
               vob[::lin,0], # 72 is for NDBC buoy, none for airport
               color='k',
               units='y',
               scale_units='y',
               scale = 1,
               headwidth = 8,
               headlength = 10,
               width = 0.1, label='observations')
ax.set_ylim(-10, 10)
ax.set_ylabel('[m/s]')
ax.legend(loc=3)
ax.text(0.02, 0.85, 'a)',
        transform=ax.transAxes, bbox=dict(facecolor='black',
         alpha=0.1), fontsize=14)


ax = axs[1]
q = ax.quiver(num2date(time_both[::lin]), 0, umod[::lin],
               vmod[::lin], # 72 is for NDBC buoy, none for airport
               color='k',
               units='y',
               scale_units='y',
               scale = 1,
               headwidth = 8,
               headlength = 10,
               width = 0.1, label='CORE-II')
ax.set_ylim(-10, 10)
ax.set_ylabel('[m/s]')
ax.legend(loc=3)
ax.text(0.02, 0.85, 'b)',
        transform=ax.transAxes, bbox=dict(facecolor='black',
         alpha=0.1), fontsize=14)

'''ax = axs[2]
q = ax.quiver(num2date(time_both[::lin]), 0, udif[::lin],
               vdif[::lin], # 72 is for NDBC buoy, none for airport
               color='k',
               units='y',
               scale_units='y',
               scale = 1,
               headwidth = 8,
               headlength = 10,
               width = 0.1, label='CORE-II-obs')
ax.set_ylim(-10, 10)
ax.set_ylabel('[m/s]')
ax.legend(loc=3)
ax.text(0.02, 0.85, 'c)',
        transform=ax.transAxes, bbox=dict(facecolor='black',
         alpha=0.1), fontsize=14)'''
fi1 = date2num(datetime.datetime.strptime('1993-01-01', '%Y-%m-%d'))
fi2 = date2num(datetime.datetime.strptime('2009-12-31', '%Y-%m-%d'))
ax.set_xlim(fi1, fi2)
plt.savefig('fig_timeseries_model_obs_comparison.pdf', bbox_inches='tight')

# Make Climatology
uob = np.ma.masked_invalid(uob.squeeze())
vob = np.ma.masked_invalid(vob.squeeze())
uclim_ob = np.reshape(uob[:24720], (206,120)).mean(1)
uclim_ob = np.reshape(uclim_ob[:204], (17,12)).mean(0)

vclim_ob = np.reshape(vob[:24720], (206,120)).mean(1)
vclim_ob = np.reshape(vclim_ob[:204], (17,12)).mean(0)


uclim_mod = np.reshape(umod[:24720], (206,120)).mean(1)
uclim_mod = np.reshape(uclim_mod[:204], (17,12)).mean(0)

vclim_mod = np.reshape(vmod[:24720], (206,120)).mean(1)
vclim_mod = np.reshape(vclim_mod[:204], (17,12)).mean(0)


lin = 10
fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10.11, 5.08))
ax = axs[0]
ax.plot(uclim_ob, '.-', label='observations')
ax.plot(uclim_mod, '.-', label='model')
ax.axhline(y=0, color='k', alpha=0.5, lw=0.5)
ax.set_title('u')
ax.legend()
ax.set_ylabel('[m/s]')
ax = axs[1]
ax.plot(vclim_ob,'.-')
ax.plot(vclim_mod,'.-')
ax.set_ylim(-8, 8)
ax.set_title('v')
fig.suptitle('Climatology wind comparison')
ax.set_ylabel('[m/s]')
ax.axhline(y=0, color='k', alpha=0.5, lw=0.5)
plt.savefig('fig_timeseries_model_obs_comparison_clim.pdf', bbox_inches='tight')
