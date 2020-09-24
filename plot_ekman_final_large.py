import netCDF4 as S
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num, num2date, datestr2num, datetime
import sys
sys.path.append('/Users/carolina/SLOMO/codes')
sys.path.append('/Users/carolina/SLOMO/codes/python-seawater/')
import cmocean
from matplotlib import cm
from matplotlib import rcParams as rcp
import seawater as sw
from matplotlib.gridspec import GridSpec
from slomo_processing import *

rcp['axes.labelsize'] = 12.
rcp['xtick.labelsize'] = 12.
rcp['ytick.labelsize'] = 12.
rcp['lines.linewidth'] = 2.
rcp['font.family'] = 'sans serif'
rcp['xtick.top'] = True
rcp['xtick.labeltop'] = False
rcp['xtick.bottom'] = True
rcp['xtick.labelbottom'] = True

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
ulong = fileobj.variables['ULONG']
ulat = fileobj.variables['ULAT']
hu = fileobj.variables['HU'] # ocean depth at vel points
ht = fileobj.variables['HT'] # ocean depth at T points
zw = fileobj.variables['z_w'][:] / 100 # depth from surface to top of layer

fileobj = S.Dataset('../clean_data/POP_mld_1993_2009.nc', mode='r')
time = fileobj.variables['time'][:]
mld_bf = fileobj.variables['mld'][:] # in m

fileobj = S.Dataset('../clean_data/POP_vel_1993_2009.nc', mode='r')
u = fileobj.variables['u'][:] * 100
v = fileobj.variables['v'][:] * 100
lon_pop = fileobj.variables['lon'][:]
lat_pop = fileobj.variables['lat'][:]
time = fileobj.variables['time'][:]

fileobj = S.Dataset('../clean_data/POP_ssh_1993_2009.nc', mode='r')
ssh = fileobj.variables['ssh'][:] # in cm

t = num2date(time)
month = np.ones(time.shape[-1]); year = np.ones(time.shape[-1])
for ii in range(0, time.shape[-1]):
  month[ii] = t[ii].month
  year[ii] = t[ii].year

X, Y = deg_to_meter((lon_pop[:], lat_pop[:]), (lon_pop[:].min(), lat_pop[:].min()))
junk, dx = np.gradient(X)
dy, junk = np.gradient(Y)
dy_ssh = np.empty(ssh.shape)
dx_ssh = np.empty(ssh.shape)
# calculates ugeo
for ii in np.arange(0, dy_ssh.shape[0]):
  dy_ssh[ii, :, :], dx_ssh[ii, :, :] = np.gradient(ssh[ii, :, :] / 100)
  print(str(ii) + " out of " + str(dy_ssh.shape[0]))
deta_y = dy_ssh / dy
deta_x = dx_ssh / dx
g = 9.81
f = coriolis(lat_pop[:])[0]
u_pop = -g / f * deta_y
v_pop = g / f * deta_x

ssh_pop = ssh
ssh_pop = ssh_pop - ssh_pop.mean(0)
u_pop = u_pop * 100
v_pop = v_pop * 100
ugeo = u_pop - u_pop.mean(0)
vgeo = v_pop - v_pop.mean(0)
mag_geo = np.sqrt(ugeo ** 2 + vgeo ** 2)

ugeo = np.ma.masked_less(ugeo, -100)
ugeo = np.ma.masked_greater(ugeo,100)
vgeo = np.ma.masked_less(vgeo, -100)
vgeo = np.ma.masked_greater(vgeo,100)


olevels2 = np.arange(0, 3000, 5000)
olevels = np.arange(0, 36, 1)

mld_bf[mld_bf==0] = 30

junk, f1 = find_nearest(lat_pop[:,0], -2.1)
junk, f2 = find_nearest(lat_pop[:,0], 2.1)
ugeo[:, f1:f2] = np.ma.masked
vgeo[:, f1:f2] = np.ma.masked

# take out bad values close to the coast
junk, idlat = find_nearest(lat_pop[:,0], -4.85)
junk, idlon = find_nearest(lon_pop[0,:], 55.65)
ugeo[:, idlat, idlon] = np.ma.masked
vgeo[:, idlat, idlon] = np.ma.masked

junk, idlat = find_nearest(lat_pop[:,0], -4.85)
junk, idlon = find_nearest(lon_pop[0,:], 55.64)
ugeo[:, idlat, idlon] = np.ma.masked
vgeo[:, idlat, idlon] = np.ma.masked

junk, idlat = find_nearest(lat_pop[:,0], -4.65)
junk, idlon = find_nearest(lon_pop[0,:], 55.45)
ugeo[:, idlat, idlon] = np.ma.masked
vgeo[:, idlat, idlon] = np.ma.masked

axisli = [52, 58.5, -8, -3]
colori = cmocean.cm.solar_r
# colori = cm.Spectral_r


lin = 3
junk, idyt = find_nearest(lat_pop[::lin,0], 3)
junk, idyb = find_nearest(lat_pop[::lin,0], -3)
fig, axs = plt.subplots(3,4, sharex=True, sharey=True, figsize=(12.76,7.33))
tit = 'DJF'
ax = axs[0,0]
ax.set_ylabel('Total', fontsize=18)
idxa = np.where((month == 12) | (month == 1) | (month == 2))[0]
mli = mld_bf[idxa].mean(0)
mli = np.ma.masked_values(mli, 0)
index = np.ma.masked_all_like(mli)
u_li = np.ma.masked_all_like(mli)
v_li = np.ma.masked_all_like(mli)
uli = u[idxa].mean(0)
vli = v[idxa].mean(0)
for ii in range(index.shape[0]):
	for jj in range(index.shape[1]):
		u_li[ii, jj] = uli[:find_nearest(zw, mli[ii, jj])[1], ii, jj].mean(0)
		v_li[ii, jj] = vli[:find_nearest(zw, mli[ii, jj])[1], ii, jj].mean(0)
utot = u_li
vtot = v_li
junk, fi_y = find_nearest(lat_pop[:,0], -4.65)
junk, fi_x = find_nearest(lon_pop[0,:], 55.4)
magli = np.sqrt(u_li ** 2 + v_li ** 2)
# magli[fi_y:fi_y+4,fi_x-1:fi_x+3] = np.ma.masked
u_li[fi_y:fi_y+4,fi_x-1:fi_x+3] = np.ma.masked
v_li[fi_y:fi_y+4,fi_x-1:fi_x+3] = np.ma.masked
maski = magli.mask
ctop = ax.contourf(lon_pop, lat_pop,
	               magli, levels=olevels,
	                 cmap=colori, extend='both')
ax.contourf(lons, lats, etopo, [0, 5000],
	       extend='None', colors='k', alpha=0.5, vmax=50) #vimn is -200 for zoom
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=0.8)
ax.contour(lons, lats, etopo, [0],
	       extend='None', colors='k', linewidths=1)
Q = ax.quiver(lon_pop[::lin,::lin], lat_pop[::lin,::lin], u_li[::lin,::lin],
			  v_li[::lin,::lin],
		          scale=100, headlength=15,headwidth=15)
ax.set_title(tit)
ax.text(0.11, 1.19, 'northwest monsoon',
        transform=ax.transAxes, color='k', fontsize=14)
ax.set(adjustable='box-forced', aspect='equal')
ax.axis(axisli)
ax.text(0.03, 0.88, 'a)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.4), color='k', fontsize=14)
ax.set_rasterized(True)

ax = axs[1,0]
u_li = ugeo[idxa].mean(0)
v_li = vgeo[idxa].mean(0)
u_li = np.ma.masked_equal(u_li, 0)
v_li = np.ma.masked_equal(v_li, 0)
junk, fi_y = find_nearest(lat_pop[:,0], -4.65)
junk, fi_x = find_nearest(lon_pop[0,:], 55.4)
#u_li[maski == True] = np.ma.masked
#v_li[maski == True] = np.ma.masked
u_li[fi_y:fi_y+4,fi_x-1:fi_x+3] = np.ma.masked
v_li[fi_y:fi_y+4,fi_x-1:fi_x+3] = np.ma.masked
magli = np.sqrt(u_li ** 2 + v_li ** 2)
u_li = np.ma.masked_greater_equal(u_li, 50)
v_li = np.ma.masked_greater_equal(v_li, 50)
u_li = np.ma.masked_less_equal(u_li, -30)
v_li = np.ma.masked_less_equal(v_li, -30)
ugi = u_li
vgi = v_li
ctop = ax.contourf(lon_pop, lat_pop,
	               magli, levels=olevels, extend='max',
	                 cmap=colori)
ax.contourf(lons, lats, etopo, [0, 5000],
	       extend='None', colors='k', alpha=0.5, vmax=50) #vimn is -200 for zoom
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=0.8)
ax.contour(lons, lats, etopo, [0],
	       extend='None', colors='k', linewidths=1)
Q = ax.quiver(lon_pop[::lin,::lin], lat_pop[::lin,::lin], u_li[::lin,::lin],
			  v_li[::lin,::lin],
		          scale=100, headlength=15,headwidth=15)
ax.set(adjustable='box-forced', aspect='equal')
ax.axis(axisli)
ax.text(0.03, 0.88, 'b)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.4), color='k', fontsize=14)
ax.set_ylabel('Geostrophic',fontsize=18)
ax.set_rasterized(True)

ax = axs[2,0]
u_li = utot - ugi # residual
v_li = vtot - vgi
junk, fi_y = find_nearest(lat_pop[:,0], -4.65)
junk, fi_x = find_nearest(lon_pop[0,:], 55.4)
# u_li[fi_y:fi_y+4,fi_x-1:fi_x+3] = np.ma.masked
# v_li[fi_y:fi_y+4,fi_x-1:fi_x+3] = np.ma.masked
magli = np.sqrt(u_li ** 2 + v_li ** 2)
u_li = np.ma.masked_greater_equal(u_li, 50)
v_li = np.ma.masked_greater_equal(v_li, 50)
u_li = np.ma.masked_less_equal(u_li, -30)
v_li = np.ma.masked_less_equal(v_li, -30)
ctop = ax.contourf(lon_pop, lat_pop,
	               magli, levels=olevels, extend='max',
	                 cmap=colori)
ax.contourf(lons, lats, etopo, [0, 5000],
	       extend='None', colors='k', alpha=0.5, vmax=50) #vimn is -200 for zoom
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=0.8)
ax.contour(lons, lats, etopo, [0],
	       extend='None', colors='k', linewidths=1)
Q = ax.quiver(lon_pop[::lin,::lin], lat_pop[::lin,::lin], u_li[::lin,::lin],
			  v_li[::lin,::lin],
		          scale=100, headlength=15,headwidth=15)
ax.set(adjustable='box-forced', aspect='equal')
ax.axis(axisli)
ax.text(0.03, 0.88, 'c)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.4), color='k', fontsize=14)
ax.set_ylabel('Residual', fontsize=18)
ax.set_rasterized(True)

idxa = np.where((month == 3) | (month == 4) | (month == 5))[0]
tit = 'MAM'
ax = axs[0,1]
mli = mld_bf[idxa].mean(0)
mli = np.ma.masked_values(mli, 0)
index = np.ma.masked_all_like(mli)
u_li = np.ma.masked_all_like(mli)
v_li = np.ma.masked_all_like(mli)
uli = u[idxa].mean(0)
vli = v[idxa].mean(0)
for ii in range(index.shape[0]):
	for jj in range(index.shape[1]):
		u_li[ii, jj] = uli[:find_nearest(zw, mli[ii, jj])[1], ii, jj].mean(0)
		v_li[ii, jj] = vli[:find_nearest(zw, mli[ii, jj])[1], ii, jj].mean(0)
utot = u_li
vtot = v_li
junk, fi_y = find_nearest(lat_pop[:,0], -4.65)
junk, fi_x = find_nearest(lon_pop[0,:], 55.4)
magli = np.sqrt(u_li ** 2 + v_li ** 2)
# magli[fi_y:fi_y+4,fi_x-1:fi_x+3] = np.ma.masked
u_li[fi_y:fi_y+4,fi_x-1:fi_x+3] = np.ma.masked
v_li[fi_y:fi_y+4,fi_x-1:fi_x+3] = np.ma.masked
maski = magli.mask
ctop = ax.contourf(lon_pop, lat_pop,
	               magli, levels=olevels,
	                 cmap=colori, extend='both')
ax.contourf(lons, lats, etopo, [0, 5000],
	       extend='None', colors='k', alpha=0.5, vmax=50) #vimn is -200 for zoom
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=0.8)
ax.contour(lons, lats, etopo, [0],
	       extend='None', colors='k', linewidths=1)
Q = ax.quiver(lon_pop[::lin,::lin], lat_pop[::lin,::lin], u_li[::lin,::lin],
			  v_li[::lin,::lin],
		          scale=100, headlength=15,headwidth=15)
ax.set_title(tit)
ax.set(adjustable='box-forced', aspect='equal')
ax.axis(axisli)
ax.text(0.03, 0.88, 'd)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.4), color='k', fontsize=14)
ax.set_rasterized(True)

ax = axs[1,1]
u_li = ugeo[idxa].mean(0)
v_li = vgeo[idxa].mean(0)
u_li = np.ma.masked_equal(u_li, 0)
v_li = np.ma.masked_equal(v_li, 0)
junk, fi_y = find_nearest(lat_pop[:,0], -4.65)
junk, fi_x = find_nearest(lon_pop[0,:], 55.4)
#u_li[maski == True] = np.ma.masked
#v_li[maski == True] = np.ma.masked
u_li[fi_y:fi_y+4,fi_x-1:fi_x+3] = np.ma.masked
v_li[fi_y:fi_y+4,fi_x-1:fi_x+3] = np.ma.masked
magli = np.sqrt(u_li ** 2 + v_li ** 2)
u_li = np.ma.masked_greater_equal(u_li, 50)
v_li = np.ma.masked_greater_equal(v_li, 50)
u_li = np.ma.masked_less_equal(u_li, -30)
v_li = np.ma.masked_less_equal(v_li, -30)
ugi = u_li
vgi = v_li
ctop = ax.contourf(lon_pop, lat_pop,
	               magli, levels=olevels, extend='max',
	                 cmap=colori)
ax.contourf(lons, lats, etopo, [0, 5000],
	       extend='None', colors='k', alpha=0.5, vmax=50) #vimn is -200 for zoom
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=0.8)
ax.contour(lons, lats, etopo, [0],
	       extend='None', colors='k', linewidths=1)
Q = ax.quiver(lon_pop[::lin,::lin], lat_pop[::lin,::lin], u_li[::lin,::lin],
			  v_li[::lin,::lin],
		          scale=100, headlength=15,headwidth=15)
ax.set(adjustable='box-forced', aspect='equal')
ax.axis(axisli)
ax.text(0.03, 0.88, 'e)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.4), color='k', fontsize=14)
ax.set_rasterized(True)

ax = axs[2,1]
u_li = utot - ugi # residual
v_li = vtot - vgi
junk, fi_y = find_nearest(lat_pop[:,0], -4.65)
junk, fi_x = find_nearest(lon_pop[0,:], 55.4)
# u_li[fi_y:fi_y+4,fi_x-1:fi_x+3] = np.ma.masked
# v_li[fi_y:fi_y+4,fi_x-1:fi_x+3] = np.ma.masked
magli = np.sqrt(u_li ** 2 + v_li ** 2)
u_li = np.ma.masked_greater_equal(u_li, 50)
v_li = np.ma.masked_greater_equal(v_li, 50)
u_li = np.ma.masked_less_equal(u_li, -30)
v_li = np.ma.masked_less_equal(v_li, -30)
ctop = ax.contourf(lon_pop, lat_pop,
	               magli, levels=olevels, extend='max',
	                 cmap=colori)
ax.contourf(lons, lats, etopo, [0, 5000],
	       extend='None', colors='k', alpha=0.5, vmax=50) #vimn is -200 for zoom
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=0.8)
ax.contour(lons, lats, etopo, [0],
	       extend='None', colors='k', linewidths=1)
Q = ax.quiver(lon_pop[::lin,::lin], lat_pop[::lin,::lin], u_li[::lin,::lin],
			  v_li[::lin,::lin],
		          scale=100, headlength=15,headwidth=15)
ax.set(adjustable='box-forced', aspect='equal')
ax.axis(axisli)
ax.text(0.03, 0.88, 'f)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.4), color='k', fontsize=14)
ax.set_rasterized(True)


ax = axs[0,2]
idxa = np.where((month == 6) | (month == 7) | (month == 8))[0]
tit = 'JJA'
mli = mld_bf[idxa].mean(0)
mli = np.ma.masked_values(mli, 0)
index = np.ma.masked_all_like(mli)
u_li = np.ma.masked_all_like(mli)
v_li = np.ma.masked_all_like(mli)
uli = u[idxa].mean(0)
vli = v[idxa].mean(0)
for ii in range(index.shape[0]):
	for jj in range(index.shape[1]):
		u_li[ii, jj] = uli[:find_nearest(zw, mli[ii, jj])[1], ii, jj].mean(0)
		v_li[ii, jj] = vli[:find_nearest(zw, mli[ii, jj])[1], ii, jj].mean(0)
utot = u_li
vtot = v_li
junk, fi_y = find_nearest(lat_pop[:,0], -4.65)
junk, fi_x = find_nearest(lon_pop[0,:], 55.4)
magli = np.sqrt(u_li ** 2 + v_li ** 2)
# magli[fi_y:fi_y+4,fi_x-1:fi_x+3] = np.ma.masked
u_li[fi_y:fi_y+4,fi_x-1:fi_x+3] = np.ma.masked
v_li[fi_y:fi_y+4,fi_x-1:fi_x+3] = np.ma.masked
maski = magli.mask
ctop = ax.contourf(lon_pop, lat_pop,
	               magli, levels=olevels,
	                 cmap=colori, extend='both')
ax.contourf(lons, lats, etopo, [0, 5000],
	       extend='None', colors='k', alpha=0.5, vmax=50) #vimn is -200 for zoom
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=0.8)
ax.contour(lons, lats, etopo, [0],
	       extend='None', colors='k', linewidths=1)
Q = ax.quiver(lon_pop[::lin,::lin], lat_pop[::lin,::lin], u_li[::lin,::lin],
			  v_li[::lin,::lin],
		          scale=100, headlength=15,headwidth=15)
ax.set_title(tit)
ax.text(0.11, 1.19, 'southeast monsoon',
        transform=ax.transAxes, color='k', fontsize=14)
ax.set(adjustable='box-forced', aspect='equal')
ax.axis(axisli)
ax.text(0.03, 0.88, 'g)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.4), color='k', fontsize=14)
ax.set_rasterized(True)

ax = axs[1,2]
u_li = ugeo[idxa].mean(0)
v_li = vgeo[idxa].mean(0)
u_li = np.ma.masked_equal(u_li, 0)
v_li = np.ma.masked_equal(v_li, 0)
junk, fi_y = find_nearest(lat_pop[:,0], -4.65)
junk, fi_x = find_nearest(lon_pop[0,:], 55.4)
#u_li[maski == True] = np.ma.masked
#v_li[maski == True] = np.ma.masked
u_li[fi_y:fi_y+4,fi_x-1:fi_x+3] = np.ma.masked
v_li[fi_y:fi_y+4,fi_x-1:fi_x+3] = np.ma.masked
magli = np.sqrt(u_li ** 2 + v_li ** 2)
u_li = np.ma.masked_greater_equal(u_li, 50)
v_li = np.ma.masked_greater_equal(v_li, 50)
u_li = np.ma.masked_less_equal(u_li, -30)
v_li = np.ma.masked_less_equal(v_li, -30)
ugi = u_li
vgi = v_li
ctop = ax.contourf(lon_pop, lat_pop,
	               magli, levels=olevels, extend='max',
	                 cmap=colori)
ax.contourf(lons, lats, etopo, [0, 5000],
	       extend='None', colors='k', alpha=0.5, vmax=50) #vimn is -200 for zoom
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=0.8)
ax.contour(lons, lats, etopo, [0],
	       extend='None', colors='k', linewidths=1)
Q = ax.quiver(lon_pop[::lin,::lin], lat_pop[::lin,::lin], u_li[::lin,::lin],
			  v_li[::lin,::lin],
		          scale=100, headlength=15,headwidth=15)
ax.set(adjustable='box-forced', aspect='equal')
ax.axis(axisli)
ax.text(0.03, 0.88, 'h)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.4), color='k', fontsize=14)
ax.set_rasterized(True)

ax = axs[2,2]
u_li = utot - ugi # residual
v_li = vtot - vgi
junk, fi_y = find_nearest(lat_pop[:,0], -4.65)
junk, fi_x = find_nearest(lon_pop[0,:], 55.4)
# u_li[fi_y:fi_y+4,fi_x-1:fi_x+3] = np.ma.masked
# v_li[fi_y:fi_y+4,fi_x-1:fi_x+3] = np.ma.masked
magli = np.sqrt(u_li ** 2 + v_li ** 2)
u_li = np.ma.masked_greater_equal(u_li, 50)
v_li = np.ma.masked_greater_equal(v_li, 50)
u_li = np.ma.masked_less_equal(u_li, -30)
v_li = np.ma.masked_less_equal(v_li, -30)
ctop = ax.contourf(lon_pop, lat_pop,
	               magli, levels=olevels, extend='max',
	                 cmap=colori)
ax.contourf(lons, lats, etopo, [0, 5000],
	       extend='None', colors='k', alpha=0.5, vmax=50) #vimn is -200 for zoom
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=0.8)
ax.contour(lons, lats, etopo, [0],
	       extend='None', colors='k', linewidths=1)
Q = ax.quiver(lon_pop[::lin,::lin], lat_pop[::lin,::lin], u_li[::lin,::lin],
			  v_li[::lin,::lin],
		          scale=100, headlength=15,headwidth=15)
ax.set(adjustable='box-forced', aspect='equal')
ax.axis(axisli)
ax.text(0.03, 0.88, 'i)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.4), color='k', fontsize=14)
ax.set_rasterized(True)

idxa = np.where((month == 9) | (month == 10) | (month == 11))[0]
tit = 'SON'
ax = axs[0,3]
mli = mld_bf[idxa].mean(0)
mli = np.ma.masked_values(mli, 0)
index = np.ma.masked_all_like(mli)
u_li = np.ma.masked_all_like(mli)
v_li = np.ma.masked_all_like(mli)
uli = u[idxa].mean(0)
vli = v[idxa].mean(0)
for ii in range(index.shape[0]):
	for jj in range(index.shape[1]):
		u_li[ii, jj] = uli[:find_nearest(zw, mli[ii, jj])[1], ii, jj].mean(0)
		v_li[ii, jj] = vli[:find_nearest(zw, mli[ii, jj])[1], ii, jj].mean(0)
utot = u_li
vtot = v_li
junk, fi_y = find_nearest(lat_pop[:,0], -4.65)
junk, fi_x = find_nearest(lon_pop[0,:], 55.4)
magli = np.sqrt(u_li ** 2 + v_li ** 2)
# magli[fi_y:fi_y+4,fi_x-1:fi_x+3] = np.ma.masked
u_li[fi_y:fi_y+4,fi_x-1:fi_x+3] = np.ma.masked
v_li[fi_y:fi_y+4,fi_x-1:fi_x+3] = np.ma.masked
maski = magli.mask
ctop = ax.contourf(lon_pop, lat_pop,
	               magli, levels=olevels,
	                 cmap=colori, extend='both')
ax.contourf(lons, lats, etopo, [0, 5000],
	       extend='None', colors='k', alpha=0.5, vmax=50) #vimn is -200 for zoom
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=0.8)
ax.contour(lons, lats, etopo, [0],
	       extend='None', colors='k', linewidths=1)
Q = ax.quiver(lon_pop[::lin,::lin], lat_pop[::lin,::lin], u_li[::lin,::lin],
			  v_li[::lin,::lin],
		          scale=100, headlength=15,headwidth=15)
ax.set_title(tit)
ax.set(adjustable='box-forced', aspect='equal')
ax.axis(axisli)
ax.text(0.03, 0.88, 'j)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.4), color='k', fontsize=14)
ax.set_rasterized(True)

ax = axs[1,3]
u_li = ugeo[idxa].mean(0)
v_li = vgeo[idxa].mean(0)
u_li = np.ma.masked_equal(u_li, 0)
v_li = np.ma.masked_equal(v_li, 0)
junk, fi_y = find_nearest(lat_pop[:,0], -4.65)
junk, fi_x = find_nearest(lon_pop[0,:], 55.4)
#u_li[maski == True] = np.ma.masked
#v_li[maski == True] = np.ma.masked
u_li[fi_y:fi_y+4,fi_x-1:fi_x+3] = np.ma.masked
v_li[fi_y:fi_y+4,fi_x-1:fi_x+3] = np.ma.masked
magli = np.sqrt(u_li ** 2 + v_li ** 2)
u_li = np.ma.masked_greater_equal(u_li, 50)
v_li = np.ma.masked_greater_equal(v_li, 50)
u_li = np.ma.masked_less_equal(u_li, -30)
v_li = np.ma.masked_less_equal(v_li, -30)
ugi = u_li
vgi = v_li
ctop = ax.contourf(lon_pop, lat_pop,
	               magli, levels=olevels, extend='max',
	                 cmap=colori)
ax.contourf(lons, lats, etopo, [0, 5000],
	       extend='None', colors='k', alpha=0.5, vmax=50) #vimn is -200 for zoom
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=0.8)
ax.contour(lons, lats, etopo, [0],
	       extend='None', colors='k', linewidths=1)
Q = ax.quiver(lon_pop[::lin,::lin], lat_pop[::lin,::lin], u_li[::lin,::lin],
			  v_li[::lin,::lin],
		          scale=100, headlength=15,headwidth=15)
ax.set(adjustable='box-forced', aspect='equal')
ax.axis(axisli)
ax.text(0.03, 0.88, 'k)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.4), color='k', fontsize=14)
ax.set_rasterized(True)

ax = axs[2,3]
u_li = utot - ugi # residual
v_li = vtot - vgi
junk, fi_y = find_nearest(lat_pop[:,0], -4.65)
junk, fi_x = find_nearest(lon_pop[0,:], 55.4)
# u_li[fi_y:fi_y+4,fi_x-1:fi_x+3] = np.ma.masked
# v_li[fi_y:fi_y+4,fi_x-1:fi_x+3] = np.ma.masked
magli = np.sqrt(u_li ** 2 + v_li ** 2)
u_li = np.ma.masked_greater_equal(u_li, 50)
v_li = np.ma.masked_greater_equal(v_li, 50)
u_li = np.ma.masked_less_equal(u_li, -30)
v_li = np.ma.masked_less_equal(v_li, -30)
ctop = ax.contourf(lon_pop, lat_pop,
	               magli, levels=olevels, extend='max',
	                 cmap=colori)
ax.contourf(lons, lats, etopo, [0, 5000],
	       extend='None', colors='k', alpha=0.5, vmax=50) #vimn is -200 for zoom
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=0.8)
ax.contour(lons, lats, etopo, [0],
	       extend='None', colors='k', linewidths=1)
Q = ax.quiver(lon_pop[::lin,::lin], lat_pop[::lin,::lin], u_li[::lin,::lin],
			  v_li[::lin,::lin],
		          scale=100, headlength=15,headwidth=15)
ax.set(adjustable='box-forced', aspect='equal')
ax.axis(axisli)
ax.text(0.03, 0.88, 'l)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.4), color='k', fontsize=14)
ax.set_rasterized(True)
cbar_ax = fig.add_axes([0.08, 0.05, 0.85, 0.01])
cbar = fig.colorbar(ctop, cax=cbar_ax,
	                ticks=np.arange(0,110,10), orientation='horizontal', extend='max')
cbar.set_label('[cm/s]', labelpad=-1.5)
fig.subplots_adjust(left=0.08, right=0.92, wspace=0.02, hspace=0.04)
qk = plt.quiverkey(Q, 0.90, 0.07, 25, r'$25 \frac{cm}{s}$', labelpos='E',
                   coordinates='figure')
ax.axis(axisli)
ax.set_rasterized(True)
plt.savefig('figures/figure_5.pdf',bbox_inches='tight')
