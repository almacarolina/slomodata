# reads netcdf POP data
# variables are
import netCDF4 as S
from matplotlib.dates import date2num, num2date, datestr2num, datetime
import sys
sys.path.append('/Users/carolina/SLOMO/codes')
import matplotlib.pyplot as plt
from read_pop_variable import *
from ocean_process import *
from mpl_toolkits.basemap import Basemap
from matplotlib import cm
import math
from matplotlib import rcParams as rcp
import air_sea as ais
from matplotlib.ticker import FormatStrFormatter
from slomo_processing import *


rcp['axes.labelsize'] = 14.
rcp['xtick.labelsize'] = 14.
rcp['ytick.labelsize'] = 14.
rcp['lines.linewidth'] = 2.
rcp['font.family'] = 'sans serif'

# First we get the static variables
fileobj = S.Dataset('sce_static.nc', mode='r') # reads variable
ulong_h = fileobj.variables['ULONG'][:]
ulat_h = fileobj.variables['ULAT'][:]
hu = fileobj.variables['HU'] # ocean depth at vel points
ht = fileobj.variables['HT'] # ocean depth at T points
zw = fileobj.variables['z_w'][:] / 100 # depth from surface to top of layer
levels = np.arange(0, 1000, 10)
levels_h = np.arange(0, 1000, 100)
llevels = np.arange(0, 5000, 1000)


fileobj = S.Dataset('../clean_data/POP_windstress_1993_2009.nc', mode='r')
lon_tau = fileobj.variables['lon'][:]
lat_tau = fileobj.variables['lat'][:]
time = fileobj.variables['time'][:]
taux = fileobj.variables['taux'][:] # * 1e4) / 1e5
tauy = fileobj.variables['tauy'][:] # * 1e4) / 1e5

fileobj = S.Dataset('../clean_data/POP_ssh_1993_2009.nc', mode='r')
lon_pop = fileobj.variables['lon'][:]
lat_pop = fileobj.variables['lat'][:]
timetot = fileobj.variables['time'][:]
ssh_pop = fileobj.variables['ssh'][:] # in cm
ssh_pop = np.ma.masked_greater_equal(ssh_pop, 1e10)

# mat = np.load('pd_monthly_POP_1959_2009.npz')
# pd = np.ma.masked_greater(mat['vvel'], 1e10)
# pd_sur = pd[:, 0] # get surface density
# pd_mean = pd.mean(1) # depth average density

t = num2date(time)
month = np.ones(time.shape[-1])
year = np.ones(time.shape[-1])
for ii in range(0, time.shape[-1]):
	month[ii] = t[ii].month
	year[ii] = t[ii].year

# compute Wind Stress Curl
#ulongtot, ulattot = np.meshgrid(lon_tau, lat_tau)
f, beta = coriolis(lat_tau[:])
X, Y = deg_to_meter((lon_tau[:], lat_tau[:]), (lon_tau[:].min(), lat_tau[:].min()))
junk, dx = np.gradient(X)
dy, junk = np.gradient(Y)
dy_taux = np.empty(taux.shape)
dx_tauy = np.empty(taux.shape)
# calculates ugeo
for ii in np.arange(0, dy_taux.shape[0]):
  dy_taux[ii, :, :], junk = np.gradient(taux[ii, :, :])
  junk, dx_tauy[ii, :, :] = np.gradient(tauy[ii, :, :])
  print(str(ii) + " out of " + str(dy_taux.shape[0]))
deta_x = dx_tauy / dx
deta_y = dy_taux / dy
curl = deta_x - deta_y
rho = 1027
we1 = (1 / (rho * f[None]) ) * curl
we2 = (beta[None] * taux) / (f[None] ** 2 * rho)
we = we1 + we2
taux = np.ma.masked_greater(taux, 1000)
tauy = np.ma.masked_less(tauy, -1000)

# this takes out the rho outside
we1 = (1 / (f[None]) ) * curl
we2 = (beta[None] * taux) / (f[None] ** 2)
we = we1 + we2
# dnek_dt = ((pd_sur - pd_mean) / rho)  * we
#g_p = ((pd_sur - pd_mean) / rho ** 2)
# dnek_dt = (g_p  * we)
g_p = 0.03 # m s**-2
g = 9.81 # m s **-2
dnek_dt = - (g_p  * we) / (rho * g)
dnek_dt = np.ma.masked_greater(dnek_dt, 100)
dnek_dt = np.ma.masked_less(dnek_dt, -100)

# get the ssh by integrating all the points in space times the month in second
month_sec = np.asarray([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]) * 86400
month_all = np.tile(month_sec, 17)
ssh_ek_all = dnek_dt * month_all[:, None, None]

# Select area to make Climatology
junk, idxl = find_nearest(lon_tau[0], 54)
junk, idxr = find_nearest(lat_tau[:,0], 57)
junk, idy1  = find_nearest(lat_tau[:,0], -5.5)
junk, idy2  = find_nearest(lat_tau[:,0], -4)
ssh_month_all = np.ones((12, ssh_ek_all.shape[1], ssh_ek_all.shape[2]))

for ii in range(1,13):
    ix = np.where(month == ii)[0]
    for jj in range(deta_x.shape[1]):
        for kk in range(deta_x.shape[2]):
            #if ~ssh_pop[ix, jj, kk].mask:
            ssh_month_all[ii-1, jj, kk] = ssh_pop[ix, jj, kk].mean(0)


test = np.zeros((13, ssh_month_all.shape[1], ssh_month_all.shape[2]))
test[0] = ssh_month_all[-1]
test[1:] = ssh_month_all
ssh_pop_difi = np.diff(test, axis=0)

lat = -4.669
lon =  55.646
junk, idx = find_nearest(lon_tau[0], lon)
junk, idy  = find_nearest(lat_tau[:,0], lat)

# SAVE pop Climatology
np.savez('ssh_clim_pop_19932009_point.npz',
      ssh_clim=ssh_month_all[:, idy, idx])

# SAVE pop Climatology
np.savez('ssh_clim_pop_19932009.npz',
      ssh_clim=ssh_month_all[:, idy1:idy2+1, idxl:idxr+1].mean(-1).mean(-1))


# Select Area to make Figure 11
junk, idxl = find_nearest(lon_tau[0], 54)
junk, idxr = find_nearest(lat_tau[:,0], 56.5)
junk, idy1  = find_nearest(lat_tau[:,0], -5.5)
junk, idy2  = find_nearest(lat_tau[:,0], -3.5)
ssh_month_all = np.ones((12, ssh_ek_all.shape[1], ssh_ek_all.shape[2]))
sshek_month_all = np.ones((12, ssh_ek_all.shape[1], ssh_ek_all.shape[2]))

for ii in range(1,13):
    ix = np.where(month == ii)[0]
    for jj in range(deta_x.shape[1]):
        for kk in range(deta_x.shape[2]):
            #if ~ssh_pop[ix, jj, kk].mask:
            sshek_month_all[ii-1, jj, kk] = ssh_ek_all[ix, jj, kk].mean(0)
            ssh_month_all[ii-1, jj, kk] = ssh_pop[ix, jj, kk].mean(0)


# Total contribution to SSH
test = np.zeros((13, ssh_month_all.shape[1], ssh_month_all.shape[2]))
test[0] = ssh_month_all[-1]
test[1:] = ssh_month_all
ssh_pop_difi = np.diff(test, axis=0)
ssh_pop_difi = np.ma.masked_invalid(ssh_pop_difi)

# Ekman contribution to SSH
sshek_month_all = np.ma.masked_invalid(sshek_month_all)


# Integrate dnek_dt to get the 12-month climatology
# we need to multiply for t, and each t has different seconds
month_sec = np.asarray([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]) * 86400
ssh_res = ssh_pop_difi - sshek_month_all * 100

we1 = (1 / (rho * f[None]) ) * curl
we2 = (beta[None] * taux) / (f[None] ** 2 * rho)
we = we1 + we2
wemonth_si = np.ones(12)
wemonth1_si = np.ones(12)
wemonth2_si = np.ones(12)
lon1 = 54
lon2 = 56.5
lat1 = -5.5
lat2 = -3.5
junk, idxl = find_nearest(lon_tau[0, :], lon1)
junk, idxr = find_nearest(lon_tau[0, :], lon2)
junk, idy1  = find_nearest(lat_tau[:,0], lat1)
junk, idy2  = find_nearest(lat_tau[:,0], lat2)

for ii in range(1,13):
	ix = np.where(month == ii)[0]
	wemonth_si[ii-1] = np.nanmean(we[ix, idy1:idy2+1, idxl:idxr+1].mean(-1).mean(-1),0)
	wemonth1_si[ii-1] = np.nanmean(we1[ix, idy1:idy2+1, idxl:idxr+1].mean(-1).mean(-1),0)
	wemonth2_si[ii-1] = np.nanmean(we2[ix, idy1:idy2+1, idxl:idxr+1].mean(-1).mean(-1),0)

fig, axs = plt.subplots(2, 1, figsize=(8.29,6.27))
ax = axs[0]
test = np.zeros(13)
test[0] = wemonth_si[-1]
test[1:] = wemonth_si
ax.plot(test, '.-',label=r'Ekman $\nabla \times \frac{\tau}{\rho_o f}$',ms=10)
test[0] = wemonth1_si[-1]
test[1:] = wemonth1_si
ax.plot(test, '.-', label=r'Curl $\frac{1}{\rho_o f}\nabla \times \tau$',ms=10)
test[0] = wemonth2_si[-1]
test[1:] = wemonth2_si
ax.plot(test, '.-', label=r'Beta $\frac{\beta \tau^x}{\rho_o f^2}$',ms=10)
ax.axhline(y=0, color='k', lw=0.8)
ax.legend(frameon=False, bbox_to_anchor=(0, 1), loc='upper left',)
ax.set_ylim(-0.85e-5, 1.4e-5)
ax.set_yticks(np.arange(-0.8e-5,1.4e-5,0.4e-5))
ax.set_yticklabels(['-8','-4','0','4','8','12','16'])
ax.set_ylabel(r'Ekman pumping speed [10$^{-6}$ ms$^{-1}$]')
ax.set_xticks(np.arange(0,14,1))
ax.set_xticklabels(('Dec','Jan', 'Feb', 'Mar', 'Apr','May', 'Jun','Jul', 'Aug', 'Sep','Oct',
  'Nov','Dec'))
ax.axvline(x=3, color='k', linestyle='--', lw=1, alpha=0.3)
ax.axvline(x=6, color='k', linestyle='--', lw=1, alpha=0.3)
ax.axvline(x=9, color='k', linestyle='--', lw=1, alpha=0.3)
ax.set_xlim(0,12)
ax.text(1.30, 16e-6, 'NW')
ax.text(1.1-0.22, 14.5e-6, 'monsoon')
ax.text(7.32, 16e-6, 'SE')
ax.text(7.35-0.46, 14.5e-6, 'monsoon')
ax.set_xticklabels([])
ax.text(0.01, 1.03, 'a)',
        transform=ax.transAxes, color='k', fontsize=14)

ax = axs[1]
test = np.zeros(13)
test[0] = ssh_pop_difi[-1, idy1:idy2+1, idxl:idxr+1].mean(-1).mean(-1)
test[1:] = ssh_pop_difi[:, idy1:idy2+1, idxl:idxr+1].mean(-1).mean(-1)
test[1:] = test[0:-1]
test[0] = test[-1]
ax.plot(test, '.-',label=r'$\eta_{tot}$',ms=10, color='k', alpha=0.6)
test[0] = sshek_month_all[-1, idy1:idy2+1, idxl:idxr+1].mean(-1).mean(-1)
test[1:] = sshek_month_all[:, idy1:idy2+1, idxl:idxr+1].mean(-1).mean(-1)
test[1:] = test[0:-1]
test[0] = test[-1]
ax.plot(test * 100, '.-',label=r'$\eta_{we}$',ms=10, color='C0')
ssh_res = ssh_pop_difi[:, idy1:idy2+1, idxl:idxr+1].mean(-1).mean(-1) - sshek_month_all[:, idy1:idy2+1, idxl:idxr+1].mean(-1).mean(-1)*100
test[0] = ssh_res[-1]
test[1:] = ssh_res
test[1:] = test[0:-1]
test[0] = test[-1]
ax.plot(test, '.-',label=r'$\eta_{tot} - \eta_{we}$',ms=10, color='orchid')
ax.axhline(y=0, color='k', lw=0.8)
ax.set_ylim(-11,11)
ax.set_ylabel('SSH [cm]')
ax.legend(frameon=False, bbox_to_anchor=(0, 1.05), loc='upper left',)
ax.set_xticks(np.arange(0,14,1))
ax.set_xticklabels(('Dec','Jan', 'Feb', 'Mar', 'Apr','May', 'Jun','Jul', 'Aug', 'Sep','Oct',
  'Nov','Dec'))
ax.axvline(x=3, color='k', linestyle='--', lw=1, alpha=0.3)
ax.axvline(x=6, color='k', linestyle='--', lw=1, alpha=0.3)
ax.axvline(x=9, color='k', linestyle='--', lw=1, alpha=0.3)
# ax.axvspan(0, 3, alpha=0.2, color='black')
# ax.axvspan(6, 9, alpha=0.2, color='black')
#ax.axvspan(11, 12, alpha=0.2, color='black')
ax.set_xlim(0,12)
ax.text(0.01, 1.03, 'b)',
        transform=ax.transAxes, color='k', fontsize=14)
# plt.savefig('figures/ekpump_termsPOP_wssh.pdf',bbox_inches='tight')
