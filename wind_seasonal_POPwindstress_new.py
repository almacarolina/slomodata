# reads netcdf POP data
import sys
sys.path.append('/Users/carolina/SLOMO/codes')
import netCDF4 as S
from matplotlib.dates import date2num, num2date, datestr2num, datetime
import matplotlib.pyplot as plt
from ocean_process import *
from mpl_toolkits.basemap import Basemap
from matplotlib import cm
import math
from matplotlib import rcParams as rcp
import air_sea as ais
from matplotlib.ticker import FormatStrFormatter
from slomo_processing import *
import cmocean

rcp['font.size'] = 18
rcp['axes.labelsize'] = 16.
rcp['xtick.labelsize'] = 16.
rcp['ytick.labelsize'] = 16.
rcp['lines.linewidth'] = 2.
rcp['font.family'] = 'sans serif'


# ETOPO bathymetry
# load the bathymetry
etopo1name = '/Users/carolina/SLOMO/codes/etopo1.asc'
topo_file = open(etopo1name, 'r')
# Read header (number of columns and rows, cell-size, and lower left
# coordinates)
ncols = int(topo_file.readline().split()[1])
nrows = int(topo_file.readline().split()[1])
xllcorner = float(topo_file.readline().split()[1])
yllcorner = float(topo_file.readline().split()[1])
cellsize = float(topo_file.readline().split()[1])
topo_file.close()
# Read in topography as a whole, disregarding first five rows (header)
etopo = np.loadtxt(etopo1name, skiprows=5)
# Data resolution is quite high. I decrease the data resolution
# to decrease the size of the final figure
dres = 1
# Swap the rows
etopo[:nrows+1, :] = etopo[nrows+1::-1, :]
etopo = etopo[::dres, ::dres]
# Create longitude and latitude vectors for etopo
lons = np.arange(xllcorner, xllcorner + cellsize * ncols, cellsize)[::dres]
lats = np.arange(yllcorner, yllcorner + cellsize * nrows, cellsize)[::dres]



# First we get the static variables
fileobj = S.Dataset('sce_static.nc', mode='r') # reads variable
ulong_h = fileobj.variables['ULONG']
ulat_h = fileobj.variables['ULAT']
hu = fileobj.variables['HU'] # ocean depth at vel points
ht = fileobj.variables['HT'] # ocean depth at T points
zw = fileobj.variables['z_w'][:] / 100 # depth from surface to top of layer

levels = np.arange(0, 1000, 10)
levels_h = np.arange(0, 1000, 100)
llevels = np.arange(0, 5000, 1000)
levels_ml = np.arange(-110, 1, 1)
levels_sat = np.arange(-10,12,2)
colori = cm.jet

li = 1
fileobj = S.Dataset('../clean_data/POP_windstress_1993_2009.nc', mode='r')
lon = fileobj.variables['lon'][:]
lat = fileobj.variables['lat'][:]
time = fileobj.variables['time'][:]
taux = fileobj.variables['taux'][:] # * 1e4) / 1e5
tauy = fileobj.variables['tauy'][:] # * 1e4) / 1e5


t = num2date(time)
month = np.ones(time.shape[-1])
year = np.ones(time.shape[-1])
for ii in range(0, time.shape[-1]):
	month[ii] = t[ii].month
	year[ii] = t[ii].year

# compute Wind Stress Curl
#ulongtot, ulattot = np.meshgrid(lon, lat)
f, beta = coriolis(lat[:])
X, Y = deg_to_meter((lon[:], lat[:]), (lon[:].min(), lat[:].min()))
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
# figure properties
colori = cm.bwr
colori2 = cm.Spectral_r
nimi = -.2e-6
nimax = .2e-6
levcurl = np.arange(-2e-7,2.1e-7,0.1e-7)
levwe = np.arange(-1e-5,1e-5,0.1e-5)

junk, f1 = find_nearest(lat[:,0], -2.1)
junk, f2 = find_nearest(lat[:,0], 2.1)
we[:, f1:f2] = np.ma.masked
we[:, f1:f2] = np.ma.masked
we = np.ma.masked_equal(we, 0)

fileobj = S.Dataset('../clean_data/POP_mld_1993_2009.nc', mode='r')
time = fileobj.variables['time'][:]
mld_bf = fileobj.variables['mld'][:] # in m
mld_bf = mld_bf * -1

fileobj = S.Dataset('../clean_data/POP_ssh_1993_2009.nc', mode='r')
lon_pop = fileobj.variables['lon'][:]
lat_pop = fileobj.variables['lat'][:]
ssh = fileobj.variables['ssh'][:] # in cm
ssh = np.ma.masked_greater_equal(ssh, 1e10)
spat_mean = ssh.reshape(ssh.shape[0], 256 * 401).mean(1)
ssh = ssh - spat_mean[:, None, None]
ssh = np.ma.masked_greater_equal(ssh, 1e36)
ssh = ssh - ssh.mean(0)

li = 16
fig, axs = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(14.1,10))
nombre = 'DJF'
idxa = np.where((month == 12) | (month == 1) | (month == 2))[0]
ax = axs[0,0]
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=1)
ax.contour(lon, lat, curl[idxa, :, :].mean(0), [0], colors='k', linewidths=0.5)
ctop = ax.contourf(lon, lat, curl[idxa, :, :].mean(0),
                  cmap=colori, vmin=nimi, vmax=nimax, levels=levcurl, extend='both')
#idxa = np.where((month_w == 12) | (month_w == 1) | (month_w == 2))[0]
Q = ax.quiver(lon[::li, ::li], lat[::li, ::li], taux[idxa, ::li, ::li].mean(0),
			  tauy[idxa, ::li, ::li].mean(0),
		      scale=0.8, headlength=8, headaxislength=8,  headwidth=10)
plt.axis([40, 70, -10, 4])
ax.set_title(str(nombre))
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
ax.set_rasterized(True)
plt.subplots_adjust(hspace=1e-4)
ax.set_xticks(np.arange(40,80,10))
ax.set_xticklabels((r'$40^o$E', r'$50^o$E', r'$60^o$E', r'$70^o$E'))
ax.set_yticks(np.arange(-20,10,10))
ax.set_yticklabels((r'$20^o$S', r'$10^o$S', r'$0^o$S'))
ax.text(0.03, 0.88, 'a)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)
ax.text(0.10, 1.18, 'northwest monsoon',
        transform=ax.transAxes, color='k', fontsize=16)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')

ax = axs[1,0]
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=1)
ax.contour(lon, lat, we[idxa, :, :].mean(0), [0], colors='k', linewidths=0.5)
ctop = ax.contourf(lon, lat, we[idxa, :, :].mean(0), levels=levwe,
                  cmap=colori2, extend='both')
plt.axis([40, 70, -10, 4])
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
plt.subplots_adjust(hspace=1e-4)
ax.set_xticks(np.arange(40,80,10))
ax.set_xticklabels((r'$40^o$E', r'$50^o$E', r'$60^o$E', r'$70^o$E'))
ax.set_yticks(np.arange(-20,10,10))
ax.set_yticklabels((r'$20^o$S', r'$10^o$S', r'$0^o$S'))
ax.text(0.03, 0.88, 'e)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')

ax = axs[2,0]
idxa = np.where((month == 12) | (month == 1) | (month == 2))[0]
mld_bfi = mld_bf[idxa].mean(0)
mld_bfi = np.ma.masked_values(mld_bfi, 0)
ctop = ax.contourf(lon, lat, mld_bfi,
	               cmap=cmocean.cm.tempo_r, levels=levels_ml, vmax=10, extend='min')
for c in ctop.collections:
    c.set_edgecolor("face")
    c.set_linewidth(0.000000000001)
sati = ssh[idxa].mean(0)
CS = ax.contour(lon[0,:], lat[:,0], sati,
	       levels=levels_sat, linewidths=0.6, alpha=0.6, colors='k', extend='min')
ax.clabel(CS, CS.levels, fmt='%i', inline=True, fontsize=10)
ax.contour(lon, lat, mld_bfi * -1, [50], linewidths=1, colors='k')
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=1)
ax.set(adjustable='box-forced', aspect='equal')
ax.text(0.03, 0.88, 'i)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)

nombre = 'MAM'
idxa = np.where((month == 3) | (month == 4) | (month == 5))[0]
ax = axs[0,1]
lcs = ax.contour(lons, lats, etopo, [-200, -100, 0], colors='#606060', linewidths=1)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')
ctop = ax.contourf(lon, lat, curl[idxa, :, :].mean(0),
                  cmap=colori, vmin=nimi, vmax=nimax, levels=levcurl, extend='both')
ax.contour(lon, lat, curl[idxa, :, :].mean(0), [0], colors='k', linewidths=0.5)
#idxa = np.where((month_w == 3) | (month_w == 4) | (month_w == 5))[0]
Q = ax.quiver(lon[::li, ::li], lat[::li, ::li], taux[idxa, ::li, ::li].mean(0),
			  tauy[idxa, ::li, ::li].mean(0),
		      scale=0.8, headlength=8, headaxislength=8,  headwidth=10)
plt.axis([40, 70, -10, 4])
ax.set_title(str(nombre))
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
ax.set_rasterized(True)
plt.subplots_adjust(hspace=1e-4)
ax.set_xticks(np.arange(40,80,10))
ax.set_xticklabels((r'$40^o$E', r'$50^o$E', r'$60^o$E', r'$70^o$E'))
ax.set_yticks(np.arange(-20,10,10))
ax.set_yticklabels((r'$20^o$S', r'$10^o$S', r'$0^o$S'))
ax.text(0.03, 0.88, 'b)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')

ax = axs[1,1]
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=1)
ax.contour(lon, lat, we[idxa, :, :].mean(0), [0], colors='k', linewidths=0.5)
ctop = ax.contourf(lon, lat, we[idxa, :, :].mean(0), levels=levwe,
                  cmap=colori2, extend='both')
plt.axis([40, 70, -10, 4])
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
plt.subplots_adjust(hspace=1e-4)
ax.set_xticks(np.arange(40,80,10))
ax.set_xticklabels((r'$40^o$E', r'$50^o$E', r'$60^o$E', r'$70^o$E'))
ax.set_yticks(np.arange(-20,10,10))
ax.set_yticklabels((r'$20^o$S', r'$10^o$S', r'$0^o$S'))
ax.text(0.03, 0.88, 'f)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')

ax = axs[2,1]
mld_bfi = mld_bf[idxa].mean(0)
mld_bfi = np.ma.masked_values(mld_bfi, 0)
ctop = ax.contourf(lon, lat, mld_bfi,
	               cmap=cmocean.cm.tempo_r, levels=levels_ml, vmax=10, extend='min')
for c in ctop.collections:
    c.set_edgecolor("face")
    c.set_linewidth(0.000000000001)
sati = ssh[idxa].mean(0)
CS = ax.contour(lon[0,:], lat[:,0], sati,
	       levels=levels_sat, linewidths=0.6, alpha=0.6, colors='k', extend='min')
ax.clabel(CS, CS.levels, fmt='%i', inline=True, fontsize=10)
ax.contour(lon, lat, mld_bfi * -1, [50], linewidths=1, colors='k')
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=1)
ax.set(adjustable='box-forced', aspect='equal')
ax.text(0.03, 0.88, 'j)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)

nombre = 'JJA'
idxa = np.where((month == 6) | (month == 7) | (month == 8))[0]
ax = axs[0,2]
lcs = ax.contour(lons, lats, etopo, [-200, -100, 0], colors='#606060', linewidths=1)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')
# ctop = ax.pcolormesh(ulong, ulat, curl[idxa, :, :].mean(0),
#                  cmap=colori, vmin=nimi, vmax=nimax)
ctop = ax.contourf(lon, lat, curl[idxa, :, :].mean(0),
                  cmap=colori, vmin=nimi, vmax=nimax, levels=levcurl, extend='both')
ax.contour(lon, lat, curl[idxa, :, :].mean(0), [0], colors='k', linewidths=0.5)
# idxa = np.where((month_w == 6) | (month_w == 7) | (month_w == 8))[0]
Q = ax.quiver(lon[::li, ::li], lat[::li, ::li], taux[idxa, ::li, ::li].mean(0),
			  tauy[idxa, ::li, ::li].mean(0),
		      scale=0.8, headlength=8, headaxislength=8,  headwidth=10)
plt.axis([40, 70, -10, 4])
ax.set_title(str(nombre))
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
ax.set_rasterized(True)
plt.subplots_adjust(hspace=1e-4)
ax.set_xticks(np.arange(40,80,10))
ax.set_xticklabels((r'$40^o$E', r'$50^o$E', r'$60^o$E', r'$70^o$E'))
ax.set_yticks(np.arange(-20,10,10))
ax.set_yticklabels((r'$20^o$S', r'$10^o$S', r'$0^o$S'))
ax.text(0.03, 0.88, 'c)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)
ax.text(0.10, 1.18, 'southeast monsoon',
        transform=ax.transAxes, color='k', fontsize=16)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')

ax = axs[1,2]
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=1)
ax.contour(lon, lat, we[idxa, :, :].mean(0), [0], colors='k', linewidths=0.5)
ctop = ax.contourf(lon, lat, we[idxa, :, :].mean(0), levels=levwe,
                  cmap=colori2, extend='both')
plt.axis([40, 70, -10, 4])
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
plt.subplots_adjust(hspace=1e-4)
ax.set_xticks(np.arange(40,80,10))
ax.set_xticklabels((r'$40^o$E', r'$50^o$E', r'$60^o$E', r'$70^o$E'))
ax.set_yticks(np.arange(-20,10,10))
ax.set_yticklabels((r'$20^o$S', r'$10^o$S', r'$0^o$S'))
ax.text(0.03, 0.88, 'g)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')

ax = axs[2,2]
mld_bfi = mld_bf[idxa].mean(0)
mld_bfi = np.ma.masked_values(mld_bfi, 0)
ctop = ax.contourf(lon, lat, mld_bfi,
	               cmap=cmocean.cm.tempo_r, levels=levels_ml, vmax=10, extend='min')
for c in ctop.collections:
    c.set_edgecolor("face")
    c.set_linewidth(0.000000000001)
sati = ssh[idxa].mean(0)
CS = ax.contour(lon[0,:], lat[:,0], sati,
	       levels=levels_sat, linewidths=0.6, alpha=0.6, colors='k', extend='min')
ax.clabel(CS, CS.levels, fmt='%i', inline=True, fontsize=10)
ax.contour(lon, lat, mld_bfi * -1, [50], linewidths=1, colors='k')
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=1)
ax.set(adjustable='box-forced', aspect='equal')
ax.text(0.03, 0.88, 'k)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)

nombre = 'SON'
idxa = np.where((month == 9) | (month == 10) | (month == 11))[0]
ax = axs[0,3]
lcs = ax.contour(lons, lats, etopo, [-200, -100, 0], colors='#606060', linewidths=1)
ctop = ax.contourf(lon, lat, curl[idxa, :, :].mean(0),
                  cmap=colori, vmin=nimi, vmax=nimax, levels=levcurl, extend='both')
ax.contour(lon, lat, curl[idxa, :, :].mean(0), [0], colors='k', linewidths=0.5)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')
# idxa = np.where((month_w == 9) | (month_w == 10) | (month_w == 11))[0]
Q = ax.quiver(lon[::li, ::li], lat[::li, ::li], taux[idxa, ::li, ::li].mean(0),
			  tauy[idxa, ::li, ::li].mean(0),
		      scale=0.8, headlength=8, headaxislength=8,  headwidth=10)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')
plt.axis([40, 70, -10, 4])
ax.set_title(str(nombre))
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
ax.set_rasterized(True)
plt.subplots_adjust(hspace=1e-4)
ax.set_xticks(np.arange(40,80,10))
ax.set_xticklabels((r'$40^o$E', r'$50^o$E', r'$60^o$E', r'$70^o$E'))
ax.set_yticks(np.arange(-20,10,10))
ax.set_yticklabels((r'$20^o$S', r'$10^o$S', r'$0^o$S'))
ax.text(0.03, 0.88, 'd)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)
qk = plt.quiverkey(Q, 0.88, 0.90, 0.1, r'0.1 $Pa$', labelpos='E',
                   coordinates='figure')

# fig.suptitle('ssh clim 1993-2009')
ax1 = axs[1,3]
lcs = ax1.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=1)
ax1.contour(lon, lat, we[idxa, :, :].mean(0), [0], colors='k', linewidths=0.5)
ctop2 = ax1.contourf(lon, lat, we[idxa, :, :].mean(0), levels=levwe,
                  cmap=colori2, extend='both')
ax1.set(adjustable='box-forced', aspect='equal')
ax1.set_rasterized(True)
plt.subplots_adjust(hspace=1e-4)
ax1.set_xticks(np.arange(40,80,10))
ax1.set_xticklabels((r'$40^o$E', r'$50^o$E', r'$60^o$E', r'$70^o$E'))
ax1.set_yticks(np.arange(-20,10,10))
ax1.set_yticklabels((r'$20^o$S', r'$10^o$S', r'$0^o$S'))
ax1.text(0.03, 0.88, 'h)',
        transform=ax1.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)
lcs = ax1.contourf(lons, lats, etopo, llevels, colors='#d2b466')
plt.axis([42, 65, -18, 3])

ax2 = axs[2,3]
mld_bfi = mld_bf[idxa].mean(0)
mld_bfi = np.ma.masked_values(mld_bfi, 0)
ctop3 = ax2.contourf(lon, lat, mld_bfi,
	               cmap=cmocean.cm.tempo_r, levels=levels_ml, vmax=10, extend='min')
for c in ctop.collections:
    c.set_edgecolor("face")
    c.set_linewidth(0.000000000001)
sati = ssh[idxa].mean(0)
CS = ax2.contour(lon[0,:], lat[:,0], sati,
	       levels=levels_sat, linewidths=0.6, alpha=0.6, colors='k', extend='min')
ax2.clabel(CS, CS.levels, fmt='%i', inline=True, fontsize=10)
ax2.contour(lon, lat, mld_bfi * -1, [50], linewidths=1, colors='k')
lcs = ax2.contourf(lons, lats, etopo, llevels, colors='#d2b466')
lcs = ax2.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=1)
ax2.set(adjustable='box-forced', aspect='equal')
ax2.text(0.03, 0.88, 'l)',
        transform=ax2.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)
ax2.set(adjustable='box-forced', aspect='equal')
ax2.set_rasterized(True)

cbar_ax = fig.add_axes([0.92, 0.62, 0.02, 0.25])
cbar = fig.colorbar(ctop, ax=ax, cax=cbar_ax, extend='both', ticks=np.arange(-2e-7,2.5e-7,0.5e-7))
cbar.set_label(r'[10$^{-7}$ $Nm^{-3}$]')
cbar.set_ticklabels([-2,-1.5,1,-0.5,0,0.5,1,1.5,2])
# ax.text(1.045, 1.08, r'10$^{-7}$', transform=ax.transAxes, fontsize=14)

cbar_ax = fig.add_axes([0.92, 0.36, 0.02, 0.25])
cbar = fig.colorbar(ctop2, ax=ax1, cax=cbar_ax, extend='both', ticks=[-1e-5, -0.5e-5, 0, 0.5e-5, 1e-5])
cbar.set_label(r'[10$^{-5}$ $ms^{-1}$]')
cbar.set_ticklabels([-1,-0.5,0,0.5,1])

cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.25])
cbar = fig.colorbar(ctop3, ax=ax2, cax=cbar_ax, extend='both', ticks=np.arange(-120,20,20))
cbar.set_label(r'[m]')
#cbar.set_ticklabels([-1,-0.5,0,0.5,1])
# ax1.text(1.045, 1.08, r'10$^{-5}$', transform=ax1.transAxes, fontsize=14)

plt.savefig('/Users/carolina/Dropbox/Seychelles_Paper_Alma/figure_4.pdf', bbox_inches='tight')




li = 16
fig, axs = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(14.1,8.6))
nombre = 'DJF'
idxa = np.where((month == 12) | (month == 1) | (month == 2))[0]
ax = axs[0,0]
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=1)
ax.contour(lon, lat, curl[idxa, :, :].mean(0), [0], colors='k', linewidths=0.5)
ctop = ax.contourf(lon, lat, curl[idxa, :, :].mean(0),
                  cmap=colori, vmin=nimi, vmax=nimax, levels=levcurl, extend='both')
#idxa = np.where((month_w == 12) | (month_w == 1) | (month_w == 2))[0]
Q = ax.quiver(lon[::li, ::li], lat[::li, ::li], taux[idxa, ::li, ::li].mean(0),
			  tauy[idxa, ::li, ::li].mean(0),
		      scale=0.8, headlength=8, headaxislength=8,  headwidth=10)
plt.axis([40, 70, -10, 4])
ax.set_title(str(nombre))
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
ax.set_rasterized(True)
plt.subplots_adjust(hspace=1e-4)
ax.set_xticks(np.arange(40,80,10))
ax.set_xticklabels((r'$40^o$E', r'$50^o$E', r'$60^o$E', r'$70^o$E'))
ax.set_yticks(np.arange(-20,10,10))
ax.set_yticklabels((r'$20^o$S', r'$10^o$S', r'$0^o$S'))
ax.text(0.03, 0.88, 'a)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)
ax.text(0.15, 1.12, 'northwest monsoon',
        transform=ax.transAxes, color='k', fontsize=16)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')

ax = axs[1,0]
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=1)
ax.contour(lon, lat, we[idxa, :, :].mean(0), [0], colors='k', linewidths=0.5)
ctop = ax.contourf(lon, lat, we[idxa, :, :].mean(0), levels=levwe,
                  cmap=colori2, extend='both')
plt.axis([40, 70, -10, 4])
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
plt.subplots_adjust(hspace=1e-4)
ax.set_xticks(np.arange(40,80,10))
ax.set_xticklabels((r'$40^o$E', r'$50^o$E', r'$60^o$E', r'$70^o$E'))
ax.set_yticks(np.arange(-20,10,10))
ax.set_yticklabels((r'$20^o$S', r'$10^o$S', r'$0^o$S'))
ax.text(0.03, 0.88, 'e)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')

nombre = 'MAM'
idxa = np.where((month == 3) | (month == 4) | (month == 5))[0]
ax = axs[0,1]
lcs = ax.contour(lons, lats, etopo, [-200, -100, 0], colors='#606060', linewidths=1)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')
#ctop = ax.pcolormesh(ulong, ulat, curl[idxa, :, :].mean(0),
#                  cmap=colori, vmin=nimi, vmax=nimax)
ctop = ax.contourf(lon, lat, curl[idxa, :, :].mean(0),
                  cmap=colori, vmin=nimi, vmax=nimax, levels=levcurl, extend='both')
ax.contour(lon, lat, curl[idxa, :, :].mean(0), [0], colors='k', linewidths=0.5)
#idxa = np.where((month_w == 3) | (month_w == 4) | (month_w == 5))[0]
Q = ax.quiver(lon[::li, ::li], lat[::li, ::li], taux[idxa, ::li, ::li].mean(0),
			  tauy[idxa, ::li, ::li].mean(0),
		      scale=0.8, headlength=8, headaxislength=8,  headwidth=10)
plt.axis([40, 70, -10, 4])
ax.set_title(str(nombre))
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
ax.set_rasterized(True)
plt.subplots_adjust(hspace=1e-4)
ax.set_xticks(np.arange(40,80,10))
ax.set_xticklabels((r'$40^o$E', r'$50^o$E', r'$60^o$E', r'$70^o$E'))
ax.set_yticks(np.arange(-20,10,10))
ax.set_yticklabels((r'$20^o$S', r'$10^o$S', r'$0^o$S'))
ax.text(0.03, 0.88, 'b)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')

ax = axs[1,1]
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=1)
ax.contour(lon, lat, we[idxa, :, :].mean(0), [0], colors='k', linewidths=0.5)
ctop = ax.contourf(lon, lat, we[idxa, :, :].mean(0), levels=levwe,
                  cmap=colori2, extend='both')
plt.axis([40, 70, -10, 4])
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
plt.subplots_adjust(hspace=1e-4)
ax.set_xticks(np.arange(40,80,10))
ax.set_xticklabels((r'$40^o$E', r'$50^o$E', r'$60^o$E', r'$70^o$E'))
ax.set_yticks(np.arange(-20,10,10))
ax.set_yticklabels((r'$20^o$S', r'$10^o$S', r'$0^o$S'))
ax.text(0.03, 0.88, 'f)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')

nombre = 'JJA'
idxa = np.where((month == 6) | (month == 7) | (month == 8))[0]
ax = axs[0,2]
lcs = ax.contour(lons, lats, etopo, [-200, -100, 0], colors='#606060', linewidths=1)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')
# ctop = ax.pcolormesh(ulong, ulat, curl[idxa, :, :].mean(0),
#                  cmap=colori, vmin=nimi, vmax=nimax)
ctop = ax.contourf(lon, lat, curl[idxa, :, :].mean(0),
                  cmap=colori, vmin=nimi, vmax=nimax, levels=levcurl, extend='both')
ax.contour(lon, lat, curl[idxa, :, :].mean(0), [0], colors='k', linewidths=0.5)
# idxa = np.where((month_w == 6) | (month_w == 7) | (month_w == 8))[0]
Q = ax.quiver(lon[::li, ::li], lat[::li, ::li], taux[idxa, ::li, ::li].mean(0),
			  tauy[idxa, ::li, ::li].mean(0),
		      scale=0.8, headlength=8, headaxislength=8,  headwidth=10)
plt.axis([40, 70, -10, 4])
ax.set_title(str(nombre))
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
ax.set_rasterized(True)
plt.subplots_adjust(hspace=1e-4)
ax.set_xticks(np.arange(40,80,10))
ax.set_xticklabels((r'$40^o$E', r'$50^o$E', r'$60^o$E', r'$70^o$E'))
ax.set_yticks(np.arange(-20,10,10))
ax.set_yticklabels((r'$20^o$S', r'$10^o$S', r'$0^o$S'))
ax.text(0.03, 0.88, 'c)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)
ax.text(0.15, 1.12, 'southeast monsoon',
        transform=ax.transAxes, color='k', fontsize=16)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')

ax = axs[1,2]
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=1)
ax.contour(lon, lat, we[idxa, :, :].mean(0), [0], colors='k', linewidths=0.5)
ctop = ax.contourf(lon, lat, we[idxa, :, :].mean(0), levels=levwe,
                  cmap=colori2, extend='both')
plt.axis([40, 70, -10, 4])
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
plt.subplots_adjust(hspace=1e-4)
ax.set_xticks(np.arange(40,80,10))
ax.set_xticklabels((r'$40^o$E', r'$50^o$E', r'$60^o$E', r'$70^o$E'))
ax.set_yticks(np.arange(-20,10,10))
ax.set_yticklabels((r'$20^o$S', r'$10^o$S', r'$0^o$S'))
ax.text(0.03, 0.88, 'g)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')

nombre = 'SON'
idxa = np.where((month == 9) | (month == 10) | (month == 11))[0]
ax = axs[0,3]
lcs = ax.contour(lons, lats, etopo, [-200, -100, 0], colors='#606060', linewidths=1)
ctop = ax.contourf(lon, lat, curl[idxa, :, :].mean(0),
                  cmap=colori, vmin=nimi, vmax=nimax, levels=levcurl, extend='both')
ax.contour(lon, lat, curl[idxa, :, :].mean(0), [0], colors='k', linewidths=0.5)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')
# idxa = np.where((month_w == 9) | (month_w == 10) | (month_w == 11))[0]
Q = ax.quiver(lon[::li, ::li], lat[::li, ::li], taux[idxa, ::li, ::li].mean(0),
			  tauy[idxa, ::li, ::li].mean(0),
		      scale=0.8, headlength=8, headaxislength=8,  headwidth=10)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')
plt.axis([40, 70, -10, 4])
ax.set_title(str(nombre))
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
ax.set_rasterized(True)
plt.subplots_adjust(hspace=1e-4)
ax.set_xticks(np.arange(40,80,10))
ax.set_xticklabels((r'$40^o$E', r'$50^o$E', r'$60^o$E', r'$70^o$E'))
ax.set_yticks(np.arange(-20,10,10))
ax.set_yticklabels((r'$20^o$S', r'$10^o$S', r'$0^o$S'))
ax.text(0.03, 0.88, 'd)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)
qk = plt.quiverkey(Q, 0.88, 0.85, 0.1, r'0.1 $Pa$', labelpos='E',
                   coordinates='figure')

# fig.suptitle('ssh clim 1993-2009')
ax1 = axs[1,3]
lcs = ax1.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=1)
ax1.contour(lon, lat, we[idxa, :, :].mean(0), [0], colors='k', linewidths=0.5)
ctop2 = ax1.contourf(lon, lat, we[idxa, :, :].mean(0), levels=levwe,
                  cmap=colori2, extend='both')
ax1.set(adjustable='box-forced', aspect='equal')
ax1.set_rasterized(True)
plt.subplots_adjust(hspace=1e-4)
ax1.set_xticks(np.arange(40,80,10))
ax1.set_xticklabels((r'$40^o$E', r'$50^o$E', r'$60^o$E', r'$70^o$E'))
ax1.set_yticks(np.arange(-20,10,10))
ax1.set_yticklabels((r'$20^o$S', r'$10^o$S', r'$0^o$S'))
ax1.text(0.03, 0.88, 'h)',
        transform=ax1.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)
lcs = ax1.contourf(lons, lats, etopo, llevels, colors='#d2b466')
plt.axis([42, 65, -18, 3])

cbar_ax = fig.add_axes([0.92, 0.50, 0.02, 0.40])
cbar = fig.colorbar(ctop, ax=ax, cax=cbar_ax, extend='both', ticks=np.arange(-2e-7,2.5e-7,0.5e-7))
cbar.set_label(r'[10$^{-7}$ $Nm^{-3}$]')
cbar.set_ticklabels([-2,-1.5,1,-0.5,0,0.5,1,1.5,2])
# ax.text(1.045, 1.08, r'10$^{-7}$', transform=ax.transAxes, fontsize=14)

cbar_ax = fig.add_axes([0.92, 0.08, 0.02, 0.40])
cbar = fig.colorbar(ctop2, ax=ax1, cax=cbar_ax, extend='both', ticks=[-1e-5, -0.5e-5, 0, 0.5e-5, 1e-5])
cbar.set_label(r'[10$^{-5}$ $ms^{-2}$]')
cbar.set_ticklabels([-1,-0.5,0,0.5,1])
# ax1.text(1.045, 1.08, r'10$^{-5}$', transform=ax1.transAxes, fontsize=14)

plt.savefig('figures/wsc_COREPOP_wek.pdf', bbox_inches='tight')
