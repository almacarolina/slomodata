# plot ellipses ADCP N and East
# band pass ADCP data
import sys
sys.path.append("../../codes/")
import numpy as np
from matplotlib import rcParams as rcp
import matplotlib.pyplot as plt
from matplotlib import cm as colorm
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
from matplotlib.path import Path
import matplotlib.patches as patches
import netCDF4 as S
# external codes needed
import cmocean
from matplotlib.dates import date2num, num2date, datestr2num, datetime
from slomo_processing import *

rcp['axes.labelsize'] = 12.
rcp['xtick.labelsize'] = 12.
rcp['ytick.labelsize'] = 12.
rcp['lines.linewidth'] = 2.
rcp['font.family'] = 'serif'

late = -4.669
lone =  55.646
file = S.Dataset('/Users/carolina/SLOMO/clean_data/adcp_E_slomo_hourly_20162019.nc', mode='r')
utot_e = file.variables['u'][:]
vtot_e = file.variables['v'][:]
time_e = file.variables['time_vel'][:]
timetot = num2date(time_e)
month_e = np.ones(time_e.shape[-1])
year_e = np.ones(time_e.shape[-1])
day_e = np.ones(time_e.shape[-1])
for ii in range(0, len(timetot)):
	month_e[ii] = timetot[ii].month
	day_e[ii] = timetot[ii].day
	year_e[ii] = timetot[ii].year

latn = -3.75
lonn =  55.60187
file = S.Dataset('/Users/carolina/SLOMO/clean_data/adcp_N_slomo_hourly_20162019.nc', mode='r')
utot_n = file.variables['u'][:]
vtot_n = file.variables['v'][:]
time_n = file.variables['time'][:]
timetot = num2date(time_n)
# select specific months
month_n = np.ones(time_n.shape[-1])
year_n = np.ones(time_n.shape[-1])
day_n = np.ones(time_n.shape[-1])
for ii in range(0, time_n.shape[-1]):
	month_n[ii] = timetot[ii].month
	day_n[ii] = timetot[ii].day
	year_n[ii] = timetot[ii].year

# make ellipses
Maj, Min, theta, Ener_hf = ellvar(utot_n.mean(-1), vtot_n.mean(-1))
s = 10 # scaling parameter
t = np.arange(0, 2 * np.pi, 0.1)
a1 = Maj[0, 1]
a2 = Min[0, 1]
theta = theta[0, 1]
# theta = 1 / 2 * np.arctan2(2 * uv, uu - vv)
x0 = a1 * np.cos(t)
y0 = a2 * np.sin(t)
x = x0 * np.cos(theta) - y0 * np.sin(theta)
y = x0 * np.sin(theta) + y0 * np.cos(theta)
xn = lonn + s * x / np.cos(np.pi / 180 * latn)
yn = latn + s * y

Maj, Min, theta, Ener_hf = ellvar(utot_e.mean(-1), vtot_e.mean(-1))
t = np.arange(0, 2 * np.pi, 0.1)
a1 = Maj[0, 1]
a2 = Min[0, 1]
theta = theta[0, 1]
# theta = 1 / 2 * np.arctan2(2 * uv, uu - vv)
x0 = a1 * np.cos(t)
y0 = a2 * np.sin(t)
x = x0 * np.cos(theta) - y0 * np.sin(theta)
y = x0 * np.sin(theta) + y0 * np.cos(theta)
xe = lone + s * x / np.cos(np.pi / 180 * late)
ye = late + s * y

# plot ellipse on map
etopo1name = '../../codes/etopo1.asc'
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

lat_adcp_e = -4.669
lon_adcp_e = 55.646
lat_adcp_w = -4.718
lon_adcp_w = 55.397
lat_adcp_w1 = -4.71955
lon_adcp_w1 = 55.39661
# 4°43'15.89"S,  55°23'45.85"E.
lat_adcp_w2 = -4.7211
lon_adcp_w2 = 55.4069
lat_adcp_n = -3.75
lon_adcp_n =  55.60187
lat_adcp = np.asarray([lat_adcp_e, lat_adcp_n])
lon_adcp = np.asarray([lon_adcp_e, lon_adcp_n])
lon_met = 55.5114
lat_met = -4.6711

test = [-3000,-2000,-1500,-500, -200, -150 -100, -50, 0]
olevels = np.arange(-200, 20, 10)  # check -400 is for mascarene plateau
test = np.asarray([-5000, -4500, -4000, -3500, -3000, -2000, -1000, -100, 0])
fig = plt.figure(figsize=(13.58, 10.87))
# gs = GridSpec(1, 3, hspace=0.2, left=0.05)
gs = GridSpec(3, 6)
plt.subplot2grid((3, 6), (0,0), colspan=2, rowspan=2)
ax1 = plt.gca()
# Create basemap, 870 km east-west, 659 km north-south,
# intermediate resolution, Transverse Mercator projection,
# centred around lon/lat 1/58.5
m1 = Basemap(projection='merc', llcrnrlon=40, llcrnrlat=-20,
            urcrnrlon=70, urcrnrlat=15, fix_aspect=True, resolution='h')
# Draw coast line
m1.drawcoastlines(color='k')
# Draw continents and lakes
m1.fillcontinents(color='#d2b466')
# Draw a think border around the whole map
m1.drawmapboundary(linewidth=3)
# Convert etopo1 coordinates lon/lat in to x/y in
# (From the basemap help: Calling a Basemap class instance with the arguments
# lon, lat will convert lon/lat (in degrees) to x/y map projection coordinates
# (in meters).)
rlons, rlats = m1(*np.meshgrid(lons, lats))
# Draw etopo1, first for land and then for the ocean, with different colormaps
llevels = np.arange(0, 2000, 100)  # check etopo.ravel().max()
lcs = m1.contourf(rlons, rlats, etopo, llevels,
                cmap=colorm.terrain)
#olevels = [-5000,
#           -1000, -500, -300, -200, -150, -100, -50, -10, 0]
cso = m1.contourf(rlons, rlats, etopo, test, extend='min',
                 #levels=[-1000,
                 #-500, -300, -200, -150, -100, -50, -20, 0],
                 cmap=cmocean.cm.deep_r, vmax=0, vmin=-5000) #vimn is -200 for zoom
for c in cso.collections:
    c.set_edgecolor("face")
    c.set_linewidth(0.000000000001)
# m.plot(rx[0,:], ry[:,1],'or',ms=14)
m1.drawmeridians(np.arange(45, 90, 5), labels=[0, 0, 0, 1],
                fontsize=12, linewidth=0)
m1.drawparallels(np.arange(-20, 15, 5), labels=[1, 0, 0, 1],
                fontsize=12, linewidth=0)
#CSO = m.contour(rlons, rlats, etopo,
#                 levels=[-10000, -1000,
#                 -500, -50], colors='k')
#plt.clabel(cso, inline=1, fontsize=10, fmt='%1i')
ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("bottom", size="5%",pad=0.25)
cbar = plt.colorbar(cso, cax=cax, ticks=[-5000,
                    -4000, -2000, -100, 0], orientation='horizontal',
                    extendrect='True', extend='Min')
# cbar.ax.set_xticklabels([10 ,8, 6, 4, 2, 0])
cbar.set_label(r'$[m]$')
plt.text(0.03, 0.92, 'a)', bbox=dict(facecolor='white',
	alpha=0.8), transform=ax.transAxes, fontsize=18)

ax.set_rasterized(True)

test = [-3000,-2000,-1500,-500, -200, -150 -100, -50, 0]
olevels = np.arange(-150, 20, 5)  # check -400 is for mascarene plateau and 150 for zoom Mahe
test = np.asarray([-3000, -2000, -1000, -200, -100, -50, 0])
olevels2 =[-3000, -1000, -200, -100]
# fig, ax = plt.subplots(1, 1, figsize=(10, 6.26))


# gs = GridSpec(3,4)
ax3 = plt.subplot2grid((3,6), (0,2), colspan=4, rowspan=2)
# Create basemap, 870 km east-west, 659 km north-south,
# intermediate resolution, Transverse Mercator projection,
# centred around lon/lat 1/58.5
m2 = Basemap(projection='merc', llcrnrlon=54, llcrnrlat=-6.2,
            urcrnrlon=57, urcrnrlat=-3.5, fix_aspect=True, resolution='h')# 53, -5, 58, -3
# Draw coast line
m2.drawcoastlines(color='k', linewidth=1)
# Draw continents and lakes
m2.fillcontinents(color='#d2b466')
# Draw a think border around the whole map
m2.drawmapboundary(linewidth=3)
# Convert etopo1 coordinates lon/lat in to x/y in
# (From the basemap help: Calling a Basemap class instance with the arguments
# lon, lat will convert lon/lat (in degrees) to x/y map projection coordinates
# (in meters).)
rlons, rlats = m2(*np.meshgrid(lons, lats))
# rlons_ther, rlats_ther = m(*np.meshgrid(lon_ther, lat_ther))
rlons_adcpe, rlats_adcpe = m2(*np.meshgrid(lon_adcp_e, lat_adcp_e))
rlons_adcpw, rlats_adcpw = m2(*np.meshgrid(lon_adcp_w, lat_adcp_w))
rlons_adcpw1, rlats_adcpw1 = m2(*np.meshgrid(lon_adcp_w1, lat_adcp_w1))
rlons_adcpw2, rlats_adcpw2 = m2(*np.meshgrid(lon_adcp_w2, lat_adcp_w2))
rlons_adcpn, rlats_adcpn = m2(*np.meshgrid(lon_adcp_n, lat_adcp_n))
rlons_met, rlats_met = m2(*np.meshgrid(lon_met, lat_met))
# Draw etopo1, first for land and then for the ocean, with different colormaps
llevels = np.arange(0, 2000, 100)  # check etopo.ravel().max()
# m.plot(rlons_adcp[0, :], rlats_adcp[:, 0], '^', color='#a44d1a', ms=8, label='ADCP')
m2.plot(rlons_adcpn, rlats_adcpn, '^', color='magenta', ms=17, label='ADCPN')
m2.plot(rlons_adcpe, rlats_adcpe+2300, '^', color='k', ms=17, label='ADCPE')
m2.plot(rlons_adcpw, rlats_adcpw, 'o', color='darkorange', ms=11, label='TChW')
m2.plot(rlons_adcpe, rlats_adcpe, 'o', color='k', ms=13, label='TChE',markeredgecolor='w')

axi = m2.contour(rlons, rlats, etopo, [-1000],
    linewidths=0.8, colors='k', linestyles='solid') #vimn is -200 for zoom
ax3.clabel(axi, axi.levels[::2], inline=True, fmt='%i', fontsize=10)
axi = m2.contour(rlons, rlats, etopo, [-100],
    linewidths=0.8, colors='k', linestyles='solid', alpha=0.8) #vimn is -200 for zoom #006abc
ax3.clabel(axi, axi.levels[::2], inline=True, fmt='%i', fontsize=10)
axi = m2.contour(rlons, rlats, etopo, [-30],
    linewidths=0.8, colors='k', linestyles='solid', alpha=0.6) #vimn is -200 for zoom #ffb6b6
ax3.clabel(axi, axi.levels[::2], inline=True, fmt='%i', fontsize=10)
m2.drawmeridians(np.arange(54, 59, 1), labels=[0, 0, 0, 1],
                fontsize=14, linewidth=0)
m2.drawparallels(np.arange(-3.5, -7, -0.5), labels=[1, 0, 0, 1],
                fontsize=14, linewidth=0)

Maj, Min, theta, Ener_hf = ellvar(utot_n.mean(-1), vtot_n.mean(-1))
s = 10 # scaling parameter
t = np.arange(0, 2 * np.pi, 0.1)
a1 = Maj[0, 1]
a2 = Min[0, 1]
theta = theta[0, 1]
# theta = 1 / 2 * np.arctan2(2 * uv, uu - vv)
x0 = a1 * np.cos(t)
y0 = a2 * np.sin(t)
x = x0 * np.cos(theta) - y0 * np.sin(theta)
y = x0 * np.sin(theta) + y0 * np.cos(theta)
xn = lonn + s * x / np.cos(np.pi / 180 * latn)
yn = latn + s * y
rlons_xn, rlats_yn = m2(*np.meshgrid(xn, yn))
m2.plot(rlons_xn[0, :], rlats_yn[:,0], color='#EF57CF',lw=2)#1ea1a5

idxa = np.where((month_e == 12) | (month_e == 1) | (month_e == 2))[0]
Maj, Min, theta, Ener_hf = ellvar(utot_e.mean(-1), vtot_e.mean(-1)) # #562384
t = np.arange(0, 2 * np.pi, 0.1)
a1 = Maj[0, 1]
a2 = Min[0, 1]
theta = theta[0, 1]
# theta = 1 / 2 * np.arctan2(2 * uv, uu - vv)
x0 = a1 * np.cos(t)
y0 = a2 * np.sin(t)
x = x0 * np.cos(theta) - y0 * np.sin(theta)
y = x0 * np.sin(theta) + y0 * np.cos(theta)
xe = lone + s * x / np.cos(np.pi / 180 * late)
ye = late + s * y
rlons_xe, rlats_ye = m2(*np.meshgrid(xe, ye))
m2.plot(rlons_xe[0, :], rlats_ye[:,0], color='k',lw=2) #orange


idxa = np.where((month_e == 6) | (month_e == 7) | (month_e == 8))[0]
Maj, Min, theta, Ener_hf = ellvar(utot_e[idxa, :].mean(-1), vtot_e[idxa, :].mean(-1))
t = np.arange(0, 2 * np.pi, 0.1)
a1 = Maj[0, 1]
a2 = Min[0, 1]
theta = theta[0, 1]
# theta = 1 / 2 * np.arctan2(2 * uv, uu - vv)
x0 = a1 * np.cos(t)
y0 = a2 * np.sin(t)
x = x0 * np.cos(theta) - y0 * np.sin(theta)
y = x0 * np.sin(theta) + y0 * np.cos(theta)
xe = lone + s * x / np.cos(np.pi / 180 * late)
ye = late + s * y
rlons_xe, rlats_ye = m2(*np.meshgrid(xe, ye))
# m2.plot(rlons_xe[0, :], rlats_ye[:,0], color='#562384',lw=2)
theta = 1
t = np.arange(0, 2 * np.pi, 0.1)
#t = 5.3
x0 = 0.01 * np.cos(t)
y0 = 0.01 * np.sin(t)
x = x0 * np.cos(theta) - y0 * np.sin(theta)
y = x0 * np.sin(theta) + y0 * np.cos(theta)
xf = 54.7 + s * x  / np.cos(np.pi / 180 * -5.5)
yf = -5.3 + s * y
rlons_xf, rlats_yf = m2(*np.meshgrid(xf, yf))
m2.plot(rlons_xf[0, :], rlats_yf[:,0], color='k',lw=2) # purple
plt.text(rlons_xf[0,0], rlats_yf[0,0], '10 cm/s', color='k')
# m.quiver(rlons_adcp[0, 0], rlats_adcp[0, 0], 9.5, 0,
# 	color='m', angles='xy', scale_units='xy', scale=1)
m2.plot(rlons_met, rlats_met, '*', color='green', ms=17, label='Wind Station')
plt.text(0.03, 0.92, 'b)', bbox=dict(facecolor='white',
	alpha=0.5), transform=ax3.transAxes, fontsize=18)
plt.rcParams['font.family'] = 'sans-serif'
legend = plt.legend(loc=(.03, .035), ncol=1,
                  frameon=False, handletextpad=0.01, fontsize=14, markerscale=0.8)
frame = legend.get_frame()
ax_cb =plt.gca()
# divider = make_axes_locatable(ax_cb)
# cax = divider.append_axes("bottom", size="5%",pad=0.25)
# cbar = plt.colorbar(cso, cax=cax, orientation='horizontal',
#                     extendrect='True', extend='Min')
# cbar.ax3.set_xticklabels([10 ,8, 6, 4, 2, 0])
# cbar.set_label(r'$[m]$')
ax.set_rasterized(True)
#Drawing the zoom rectangles:
lbx1, lby1 = m1(*m2(m2.xmin, m2.ymin, inverse= True))
ltx1, lty1 = m1(*m2(m2.xmin, m2.ymax, inverse= True))
rtx1, rty1 = m1(*m2(m2.xmax, m2.ymax, inverse= True))
rbx1, rby1 = m1(*m2(m2.xmax, m2.ymin, inverse= True))
verts1 = [
    (lbx1, lby1), # left, bottom
    (ltx1, lty1), # left, top
    (rtx1, rty1), # right, top
    (rbx1, rby1), # right, bottom
    (lbx1, lby1), # ignored
    ]
codes2 = [Path.MOVETO,
         Path.LINETO,
         Path.LINETO,
         Path.LINETO,
         Path.CLOSEPOLY,
         ]
path2= Path(verts1, codes2)
patch2 = patches.PathPatch(path2, facecolor='none', edgecolor='r', lw=1.5)
ax.add_patch(patch2)
# fig.subplots_adjust(hspace=0.01)
fig.subplots_adjust(wspace=0.15)
# plt.savefig('bathy_final.pdf', bbox_inches='tight')
