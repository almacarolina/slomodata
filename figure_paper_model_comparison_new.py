# make figures seasonal
import netCDF4 as S
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num, num2date, datestr2num, datetime
import cmocean
from matplotlib import cm
from matplotlib import rcParams as rcp
import sys
sys.path.append("../../codes")
from slomo_processing import *
import xesmf as xe
import xarray as xr

rcp['font.size'] = 18
rcp['axes.labelsize'] = 16.
rcp['xtick.labelsize'] = 16.
rcp['ytick.labelsize'] = 16.
rcp['lines.linewidth'] = 2.
rcp['font.family'] = 'sans serif'


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


fileobj = S.Dataset('../../clean_data/POP_ssh_1993_2009.nc', mode='r')
lon_pop = fileobj.variables['lon'][:]
lat_pop = fileobj.variables['lat'][:]
timetot = fileobj.variables['time'][:]
ssh = fileobj.variables['ssh'][:] # in cm
ssh = np.ma.masked_greater_equal(ssh, 1e10)
# remove the spatial mean
spat_mean = ssh.reshape(ssh.shape[0], 256 * 401).mean(1)
ssh = ssh - spat_mean[:, None, None]
t = num2date(timetot)
month = np.ones(timetot.shape[-1]); year = np.ones(timetot.shape[-1])
for ii in range(0, timetot.shape[-1]):
  month[ii] = t[ii].month
  year[ii] = t[ii].year

# choose time as AVISO
idx1 = np.where((month == 1) & (year == 1993))[0]
idx2 = np.where((month == 12) & (year == 2009))[-1] # data ends in 2018/06/10
ssh = ssh[idx1[0]:idx2[-1]+1, :, :]
timetot_pop = timetot[idx1[0]:idx2[-1]+1]
t = num2date(timetot_pop)
month_pop = np.ones(timetot_pop.shape[-1]); year_pop = np.ones(timetot_pop.shape[-1])
for ii in range(0, timetot_pop.shape[-1]):
  month_pop[ii] = t[ii].month
  year_pop[ii] = t[ii].year

ssh_cont = np.ma.masked_greater_equal(ssh, 1e36)
X, Y = deg_to_meter((lon_pop[:], lat_pop[:]), (lon_pop[:].min(), lat_pop[:].min()))
junk, dx = np.gradient(X)
dy, junk = np.gradient(Y)
dy_ssh = np.empty(ssh_cont.shape)
dx_ssh = np.empty(ssh_cont.shape)

# calculates ugeo
for ii in np.arange(0, dy_ssh.shape[0]):
  dy_ssh[ii, :, :], dx_ssh[ii, :, :] = np.gradient(ssh_cont[ii, :, :] / 100)
  print(str(ii) + " out of " + str(dy_ssh.shape[0]))
deta_y = dy_ssh / dy
deta_x = dx_ssh / dx
g = 9.81
f = coriolis(lat_pop[:])[0]
u_pop = -g / f * deta_y
v_pop = g / f * deta_x
u_pop = np.ma.masked_greater(u_pop,5)
v_pop = np.ma.masked_greater(v_pop,5)
u_pop = np.ma.masked_less(u_pop,-5)
v_pop = np.ma.masked_less(v_pop,-5)
ssh_pop = ssh_cont

# remove currents from POP close to equator
junk, idyt = find_nearest(lat_pop[:,0], 0.8)
junk, idyb = find_nearest(lat_pop[:,0], -0.8)
u_pop[:, idyb:idyt, :] = np.ma.masked
v_pop[:, idyb:idyt, :] = np.ma.masked

# get the monthly means of ssh
ssh_month_pop = np.ones((12, ssh_pop.shape[1], ssh_pop.shape[2]))
u_month_pop = np.ones((12, ssh_pop.shape[1], ssh_pop.shape[2]))
v_month_pop = np.ones((12, ssh_pop.shape[1], ssh_pop.shape[2]))
for ii in range(1, 13):
  ssh_month_pop[ii-1] = ssh_pop[np.where(month_pop == ii)[0], :, :].mean(0)
  u_month_pop[ii-1] = u_pop[np.where(month_pop == ii)[0], :, :].mean(0)
  v_month_pop[ii-1] = v_pop[np.where(month_pop == ii)[0], :, :].mean(0)

fileobj = S.Dataset('../../ssh/1993-1996_3590E_-205lat.nc', mode='r')
u0 = fileobj.variables['ugosa'][:].squeeze() # absolute geostrophic velocity, ugosa is the geostrophic velocity anomaly referenced to 1993-2012
v0 = fileobj.variables['vgosa'][:].squeeze()
ssh0 = fileobj.variables['sla'][:] # sea level anomaly
t0 = fileobj.variables['time'][:] + date2num(datetime.datetime.strptime('1950-01-01 00:00:00', '%Y-%d-%m %H:%M:%S')) # divide by 24 if in hours
lat = fileobj.variables['latitude'][:]
lon = fileobj.variables['longitude'][:]
fi = np.where((lon < lon_pop.max()) & (lon > lon_pop.min()))[0]
fi2 = np.where((lat < lat_pop.max()) & (lat > lat_pop.min()))[0]

fileobj = S.Dataset('../../ssh/1997-2000_3590E_-205lat.nc', mode='r')
u1 = fileobj.variables['ugosa'][:].squeeze() # absolute geostrophic velocity, ugosa is the geostrophic velocity anomaly referenced to 1993-2012
v1 = fileobj.variables['vgosa'][:].squeeze()
ssh1 = fileobj.variables['sla'][:] # sea level anomaly
t1 = fileobj.variables['time'][:] + date2num(datetime.datetime.strptime('1950-01-01 00:00:00', '%Y-%d-%m %H:%M:%S')) # divide by 24 if in hours

fileobj = S.Dataset('../../ssh/2001-2004_3590E_-205lat.nc', mode='r')
u2 = fileobj.variables['ugosa'][:].squeeze() # absolute geostrophic velocity, ugosa is the geostrophic velocity anomaly referenced to 1993-2012
v2 = fileobj.variables['vgosa'][:].squeeze()
ssh2 = fileobj.variables['sla'][:] # sea level anomaly
t2 = fileobj.variables['time'][:] + date2num(datetime.datetime.strptime('1950-01-01 00:00:00', '%Y-%d-%m %H:%M:%S')) # divide by 24 if in hours

fileobj = S.Dataset('../../ssh/2005-2008_3590E_-205lat.nc', mode='r')
u3 = fileobj.variables['ugosa'][:].squeeze() # absolute geostrophic velocity, ugosa is the geostrophic velocity anomaly referenced to 1993-2012
v3 = fileobj.variables['vgosa'][:].squeeze()
ssh3 = fileobj.variables['sla'][:] # sea level anomaly
t3 = fileobj.variables['time'][:] + date2num(datetime.datetime.strptime('1950-01-01 00:00:00', '%Y-%d-%m %H:%M:%S')) # divide by 24 if in hours

fileobj = S.Dataset('../../ssh/2009-2012_3590E_-205lat.nc', mode='r')
u4 = fileobj.variables['ugosa'][:].squeeze() # absolute geostrophic velocity, ugosa is the geostrophic velocity anomaly referenced to 1993-2012
v4 = fileobj.variables['vgosa'][:].squeeze()
ssh4 = fileobj.variables['sla'][:] # sea level anomaly
t4 = fileobj.variables['time'][:] + date2num(datetime.datetime.strptime('1950-01-01 00:00:00', '%Y-%d-%m %H:%M:%S')) # divide by 24 if in hours

fileobj = S.Dataset('../../ssh/2013-2016_3590E_-205lat.nc', mode='r')
u5 = fileobj.variables['ugosa'][:].squeeze() # absolute geostrophic velocity, ugosa is the geostrophic velocity anomaly referenced to 1993-2012
v5 = fileobj.variables['vgosa'][:].squeeze()
ssh5 = fileobj.variables['sla'][:] # sea level anomaly
t5 = fileobj.variables['time'][:] + date2num(datetime.datetime.strptime('1950-01-01 00:00:00', '%Y-%d-%m %H:%M:%S')) # divide by 24 if in hours

fileobj = S.Dataset('../../ssh/2017-2018_3590E_-205lat.nc', mode='r')
u6 = fileobj.variables['ugosa'][:].squeeze() # absolute geostrophic velocity, ugosa is the geostrophic velocity anomaly referenced to 1993-2012
v6 = fileobj.variables['vgosa'][:].squeeze()
ssh6 = fileobj.variables['sla'][:] # sea level anomaly
t6 = fileobj.variables['time'][:] + date2num(datetime.datetime.strptime('1950-01-01 00:00:00', '%Y-%d-%m %H:%M:%S')) # divide by 24 if in hours

u = np.concatenate((u0, u1, u2, u3, u4, u5, u6), axis=0)
v = np.concatenate((v0, v1, v2, v3, v4, v5, v6), axis=0)
ssh = np.concatenate((ssh0, ssh1, ssh2, ssh3, ssh4, ssh5, ssh6), axis=0)
t = np.concatenate((t0, t1, t2, t3, t4, t5, t6), axis=0)

ssh = np.ma.masked_less_equal(ssh, -1e9)
mag = np.sqrt(u ** 2 + v ** 2)
 # divide by 24 if in hours
size = t.size; year = np.zeros(size); month = np.zeros(size)
# choose only specific dates
for ii in np.arange(0, size):
    year[ii] = num2date(t[ii]).year
    month[ii] = num2date(t[ii]).month
u = np.ma.masked_less_equal(u, -1e3)
v = np.ma.masked_less_equal(v, -1e3)
u_sat = np.ma.masked_greater_equal(u, 1e3)
v_sat = np.ma.masked_greater_equal(v, 1e3)


# choose time
idx1 = np.where((month == 1) & (year == 1993))[0]
idx2 = np.where((month == 12) & (year == 2009))[-1] # data ends in 2018/06/10
ssh_sat = ssh[idx1[0]:idx2[-1]+1, :, :]
u_sat = u[idx1[0]:idx2[-1]+1, :, :]
v_sat = v[idx1[0]:idx2[-1]+1, :, :]
t_sat = t[idx1[0]:idx2[-1]+1]

# do everything in cm as POP
u_sat = u_sat * 100
v_sat = v_sat * 100
ssh_sat = ssh_sat * 100

size = t_sat.size; year = np.zeros(size); month = np.zeros(size)
# choose only specific dates
for ii in np.arange(0, size):
    year[ii] = num2date(t_sat[ii]).year
    month[ii] = num2date(t_sat[ii]).month

junk, idyt = find_nearest(lat, 0.8)
junk, idyb = find_nearest(lat, -0.8)
u_sat[:, idyb:idyt, :] = np.ma.masked
v_sat[:, idyb:idyt, :] = np.ma.masked

# remove spatial and temporal mean
test = ssh_sat[:, fi2[0]:fi2[-1], fi[0]:fi[-1]]
spat_mean = test.reshape(6209, 100 * 159).mean(1)
ssh_sat = ssh_sat - spat_mean[:, None, None]

test = u_sat[:, fi2[0]:fi2[-1], fi[0]:fi[-1]]
spat_mean = test.reshape(6209, 100 * 159).mean(1)
u_sat = u_sat - spat_mean[:, None, None]

test = v_sat[:, fi2[0]:fi2[-1], fi[0]:fi[-1]]
spat_mean = test.reshape(6209, 100 * 159).mean(1)
v_sat = v_sat - spat_mean[:, None, None]

ssh_sat = ssh_sat - ssh_sat.mean(0)
u_sat = u_sat - u_sat.mean(0)
v_sat = v_sat - v_sat.mean(0)
lon_sat, lat_sat = np.meshgrid(lon, lat)

# take out bad values close to the coast
junk, idlat = find_nearest(lat_pop[:,0], -4.85)
junk, idlon = find_nearest(lon_pop[0,:], 55.65)
u_pop[:, idlat, idlon] = np.ma.masked
v_pop[:, idlat, idlon] = np.ma.masked

junk, idlat = find_nearest(lat_pop[:,0], -4.85)
junk, idlon = find_nearest(lon_pop[0,:], 55.64)
u_pop[:, idlat, idlon] = np.ma.masked
v_pop[:, idlat, idlon] = np.ma.masked

junk, idlat = find_nearest(lat_pop[:,0], -4.65)
junk, idlon = find_nearest(lon_pop[0,:], 55.45)
u_pop[:, idlat, idlon] = np.ma.masked
v_pop[:, idlat, idlon] = np.ma.masked

junk, idlat = find_nearest(lat_pop[:,0], -4.75)
junk, idlon = find_nearest(lon_pop[0,:], 55.55)
u_pop[:, idlat, idlon] = np.ma.masked
v_pop[:, idlat, idlon] = np.ma.masked

junk, idlat = find_nearest(lat_pop[:,0], -4.55)
junk, idlon = find_nearest(lon_pop[0,:], 55.35)
u_pop[:, idlat, idlon] = np.ma.masked
v_pop[:, idlat, idlon] = np.ma.masked

junk, idlat = find_nearest(lat_pop[:,0], -4.66)
junk, idlon = find_nearest(lon_pop[0,:], 55.55)
u_pop[:, idlat, idlon] = np.ma.masked
v_pop[:, idlat, idlon] = np.ma.masked

junk, idlat = find_nearest(lat_pop[:,0], -4.66)
junk, idlon = find_nearest(lon_pop[0,:], 55.65)
u_pop[:, idlat, idlon] = np.ma.masked
v_pop[:, idlat, idlon] = np.ma.masked

junk, idlat = find_nearest(lat_pop[:,0], -12)
junk, idlon = find_nearest(lon_pop[0,:], 43.5)
u_pop[:, idlat-2:idlat+2, idlon-2:idlon+2] = np.ma.masked
v_pop[:, idlat-2:idlat+2, idlon-2:idlon+2] = np.ma.masked


# remove temporal mean
ssh_pop = ssh_pop - ssh_pop.mean(0)
u_pop= u_pop * 100
v_pop = v_pop * 100
u_pop = u_pop - u_pop.mean(0)
v_pop = v_pop - v_pop.mean(0)

levels_pop = np.arange(-10,10.1, 0.1)
levels_sat = levels_pop
llevels = np.arange(0, 5000, 1000)

colori = cmocean.cm.balance
colori = cm.bwr
li = 4
levels_pop = np.arange(-10,10.1, 0.1)
levels_sat = levels_pop
llevels = np.arange(0, 5000, 1000)
fig, axs  = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(15.75, 7.22))
ax = axs[0,0]
ax.set_ylabel('AVISO')
tit_fig = 'DJF'
fi = np.where((month==12) | (month==1) | (month==2))
sati = ssh_sat[fi].mean(0)
usati = u_sat[fi]
vsati = v_sat[fi]
usati = np.ma.masked_greater(usati, 50)
vsati = np.ma.masked_greater(vsati, 50)
usati = np.ma.masked_less(usati, -50)
vsati = np.ma.masked_less(vsati, -50)
ctop = ax.contourf(lon_sat, lat_sat, sati, levels_sat,
                    extend='both', cmap=colori)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=1)
ax.contour(lon_sat, lat_sat, sati, [0],
                    colors='k', lw=0.2, linestyles='solid', alpha=0.5)
Q = ax.quiver(lon_sat[::li, ::li], lat_sat[::li, ::li], usati[:, ::li, ::li].mean(0),
        vsati[:, ::li, ::li].mean(0),
              scale=150, headlength=15, headwidth=10)
for c in lcs.collections:
  c.set_linestyle('solid')
ax.set_title(tit_fig)
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
ax.text(0.03, 0.88, 'a)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)
ax.text(0.12, 1.17, 'northwest monsoon',
        transform=ax.transAxes, color='k', fontsize=16)

ax = axs[1,0]
li = 10
tit_fig = 'POP'
ax.set_ylabel('POP')
fi = np.where((month_pop==12) | (month_pop==1) | (month_pop==2))
sati = ssh_pop[fi]
usati = u_pop[fi]
vsati = v_pop[fi]
usati = np.ma.masked_greater(usati, 50)
vsati = np.ma.masked_greater(vsati, 50)
usati = np.ma.masked_less(usati, -50)
vsati = np.ma.masked_less(vsati, -50)
ctop = ax.contourf(lon_pop, lat_pop, sati.mean(0), levels_pop,
                    extend='both', cmap=colori)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=1)
ax.contour(lon_pop, lat_pop, sati.mean(0), [0],
                    colors='k', lw=0.2, linestyles='solid', alpha=0.5)
Q = ax.quiver(lon_pop[::li, ::li], lat_pop[::li, ::li], usati[:, ::li, ::li].mean(0),
        vsati[:, ::li, ::li].mean(0),
              scale=150, headlength=15, headwidth=10)
for c in lcs.collections:
  c.set_linestyle('solid')
# ax.set_title(tit_fig)
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
ax.text(0.03, 0.88, 'e)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)

ax = axs[0,2]
li = 4
tit_fig = 'JJA'
fi = np.where((month==6) | (month==7) | (month==8))
sati = ssh_sat[fi].mean(0)
usati = u_sat[fi]
vsati = v_sat[fi]
usati = np.ma.masked_greater(usati, 50)
vsati = np.ma.masked_greater(vsati, 50)
usati = np.ma.masked_less(usati, -50)
vsati = np.ma.masked_less(vsati, -50)
ctop = ax.contourf(lon_sat, lat_sat, sati, levels_sat,
                    extend='both', cmap=colori)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=1)
ax.contour(lon_sat, lat_sat, sati, [0],
                    colors='k', lw=0.2, linestyles='solid', alpha=0.5)
Q = ax.quiver(lon_sat[::li, ::li], lat_sat[::li, ::li], usati[:, ::li, ::li].mean(0),
        vsati[:, ::li, ::li].mean(0),
              scale=150, headlength=15, headwidth=10)
for c in lcs.collections:
  c.set_linestyle('solid')
ax.set_title(tit_fig)
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
ax.text(0.03, 0.88, 'c)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)
ax.text(0.12, 1.17, 'southeast monsoon',
        transform=ax.transAxes, color='k', fontsize=16)

ax = axs[1,2]
li = 10
tit_fig = 'POP'
fi = np.where((month_pop==6) | (month_pop==7) | (month_pop==8))
sati = ssh_pop[fi]
usati = u_pop[fi]
vsati = v_pop[fi]
usati = np.ma.masked_greater(usati, 50)
vsati = np.ma.masked_greater(vsati, 50)
usati = np.ma.masked_less(usati, -50)
vsati = np.ma.masked_less(vsati, -50)
ctop = ax.contourf(lon_pop, lat_pop, sati.mean(0), levels_pop,
                    extend='both', cmap=colori)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=1)
ax.contour(lon_pop, lat_pop, sati.mean(0), [0],
                    colors='k', lw=0.2, linestyles='solid', alpha=0.5)
Q = ax.quiver(lon_pop[::li, ::li], lat_pop[::li, ::li], usati[:, ::li, ::li].mean(0),
        vsati[:, ::li, ::li].mean(0),
              scale=150, headlength=15, headwidth=10)
for c in lcs.collections:
  c.set_linestyle('solid')
# ax.set_title(tit_fig)
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
ax.text(0.03, 0.88, 'g)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)


ax = axs[0,1]
li = 4
tit_fig = 'MAM'
fi = np.where((month==3) | (month==4) | (month==5))
sati = ssh_sat[fi].mean(0)
usati = u_sat[fi]
vsati = v_sat[fi]
usati = np.ma.masked_greater(usati, 50)
vsati = np.ma.masked_greater(vsati, 50)
usati = np.ma.masked_less(usati, -50)
vsati = np.ma.masked_less(vsati, -50)
ctop = ax.contourf(lon_sat, lat_sat, sati, levels_sat,
                    extend='both', cmap=colori)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=1)
ax.contour(lon_sat, lat_sat, sati, [0],
                    colors='k', lw=0.2, linestyles='solid', alpha=0.5)
Q = ax.quiver(lon_sat[::li, ::li], lat_sat[::li, ::li], usati[:, ::li, ::li].mean(0),
        vsati[:, ::li, ::li].mean(0),
              scale=150, headlength=15, headwidth=10)
for c in lcs.collections:
  c.set_linestyle('solid')
ax.set_title(tit_fig)
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
ax.text(0.03, 0.88, 'b)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)


ax = axs[1,1]
li = 10
tit_fig = 'POP'
fi = np.where((month_pop==3) | (month_pop==4) | (month_pop==5))
sati = ssh_pop[fi]
usati = u_pop[fi]
vsati = v_pop[fi]
usati = np.ma.masked_greater(usati, 50)
vsati = np.ma.masked_greater(vsati, 50)
usati = np.ma.masked_less(usati, -50)
vsati = np.ma.masked_less(vsati, -50)
ctop = ax.contourf(lon_pop, lat_pop, sati.mean(0), levels_pop,
                    extend='both', cmap=colori)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=1)
ax.contour(lon_pop, lat_pop, sati.mean(0), [0],
                    colors='k', lw=0.2, linestyles='solid', alpha=0.5)
Q = ax.quiver(lon_pop[::li, ::li], lat_pop[::li, ::li], usati[:, ::li, ::li].mean(0),
        vsati[:, ::li, ::li].mean(0),
              scale=150, headlength=15, headwidth=10)
for c in lcs.collections:
  c.set_linestyle('solid')
# ax.set_title(tit_fig)
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
ax.text(0.03, 0.88, 'f)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)


ax = axs[0,3]
li = 4
#ax.set_ylabel('AVISO')
tit_fig = 'SON'
fi = np.where((month==9) | (month==10) | (month==11))
sati = ssh_sat[fi].mean(0)
usati = u_sat[fi]
vsati = v_sat[fi]
usati = np.ma.masked_greater(usati, 50)
vsati = np.ma.masked_greater(vsati, 50)
usati = np.ma.masked_less(usati, -50)
vsati = np.ma.masked_less(vsati, -50)
ctop = ax.contourf(lon_sat, lat_sat, sati, levels_sat,
                    extend='both', cmap=colori)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=1)
ax.contour(lon_sat, lat_sat, sati, [0],
                    colors='k', lw=0.2, linestyles='solid', alpha=0.5)
Q = ax.quiver(lon_sat[::li, ::li], lat_sat[::li, ::li], usati[:, ::li, ::li].mean(0),
        vsati[:, ::li, ::li].mean(0),
              scale=150, headlength=15, headwidth=10)
for c in lcs.collections:
  c.set_linestyle('solid')
ax.set_title(tit_fig)
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
ax.text(0.03, 0.88, 'd)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)


ax = axs[1,3]
li = 10
tit_fig = 'POP'
fi = np.where((month_pop==9) | (month_pop==10) | (month_pop==11))
sati = ssh_pop[fi]
usati = u_pop[fi]
vsati = v_pop[fi]
usati = np.ma.masked_greater(usati, 50)
vsati = np.ma.masked_greater(vsati, 50)
usati = np.ma.masked_less(usati, -50)
vsati = np.ma.masked_less(vsati, -50)
ctop = ax.contourf(lon_pop, lat_pop, sati.mean(0), levels_pop,
                    extend='both', cmap=colori)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=1)
ax.contour(lon_pop, lat_pop, sati.mean(0), [0],
                    colors='k', lw=0.2, linestyles='solid', alpha=0.5)
Q = ax.quiver(lon_pop[::li, ::li], lat_pop[::li, ::li], usati[:, ::li, ::li].mean(0),
        vsati[:, ::li, ::li].mean(0),
              scale=150, headlength=15, headwidth=10)
for c in lcs.collections:
  c.set_linestyle('solid')
# ax.set_title(tit_fig)
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
ax.set_rasterized(True)
plt.subplots_adjust(hspace=1e-4)
ax.set_xticks(np.arange(40,80,10))
ax.set_xticklabels((r'$40^o$E', r'$50^o$E', r'$60^o$E', r'$70^o$E'))
ax.set_yticks(np.arange(-20,10,10))
ax.set_yticklabels((r'$20^o$S', r'$10^o$S', r'$0^o$S'))
ax.text(0.03, 0.88, 'h)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)

#plt.axis([51.63, 58.36, -8.9, -1.06])
plt.axis([42, 65, -18, 3])
# fig.suptitle('ssh clim 1993-2009')
fig.subplots_adjust(left=0.07, right=0.9, hspace=0.03, wspace=0.01)
cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
cbar = fig.colorbar(ctop, cax=cbar_ax, extend='both', ticks=np.arange(-10,12,2))
cbar.set_label('[cm]')
qk = plt.quiverkey(Q, 0.90, 0.08, 10, r'$10 \frac{cm}{s}$', labelpos='E',
                   coordinates='figure')
plt.savefig('aviso_pop_comp_seas_large.pdf', bbox_inches='tight')



# Calculate the monthly average - the 19 year average
uavg_sat = u_sat.mean(0)
vavg_sat = v_sat.mean(0)
sshavg_sat = ssh_sat.mean(0)
uavg_pop = u_pop.mean(0)
vavg_pop = v_pop.mean(0)
sshavg_pop = ssh_pop.mean(0)

colori = cmocean.cm.balance
colori = cm.bwr
li = 4
levels_pop = np.arange(-10,10.1, 0.1)
levels_sat = levels_pop
llevels = np.arange(0, 5000, 1000)
fig, axs  = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(15.75, 7.22))
ax = axs[0,0]
ax.set_ylabel('AVISO')
tit_fig = 'DJF'
fi = np.where((month==12) | (month==1) | (month==2))
sati = ssh_sat[fi].mean(0) - sshavg_sat
usati = u_sat[fi].mean(0) - uavg_sat
vsati = v_sat[fi].mean(0) - vavg_sat
usati = np.ma.masked_greater(usati, 50)
vsati = np.ma.masked_greater(vsati, 50)
usati = np.ma.masked_less(usati, -50)
vsati = np.ma.masked_less(vsati, -50)
ctop = ax.contourf(lon_sat, lat_sat, sati, levels_sat,
                    extend='both', cmap=colori)
ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')
ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=1)
ax.contour(lon_sat, lat_sat, sati, levels=np.arange(-10,12,2),
                    colors='k', linewidths=1, alpha=0.5)
ax.contour(lon_sat, lat_sat, sati, [0],
                    colors='k', linewidths=1.4, linestyles='solid', alpha=0.6)
ax.set_title(tit_fig)
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
ax.text(0.03, 0.88, 'a)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)
ax.text(0.12, 1.17, 'northwest monsoon',
        transform=ax.transAxes, color='k', fontsize=16)

plt.axis([42, 65, -18, 3])
ax = axs[1,0]
li = 10
tit_fig = 'POP'
ax.set_ylabel('POP')
fi = np.where((month_pop==12) | (month_pop==1) | (month_pop==2))
sati = ssh_pop[fi].mean(0) - ssh_pop
usati = u_pop[fi].mean(0) - uavg_pop
vsati = v_pop[fi].mean(0) - vavg_pop
usati = np.ma.masked_greater(usati, 50)
vsati = np.ma.masked_greater(vsati, 50)
usati = np.ma.masked_less(usati, -50)
vsati = np.ma.masked_less(vsati, -50)
ctop = ax.contourf(lon_pop, lat_pop, sati.mean(0), levels_pop,
                    extend='both', cmap=colori)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=1)
ax.contour(lon_pop, lat_pop, sati.mean(0), levels=np.arange(-10,12,2),
                    colors='k', linewidths=1, alpha=0.5)
ax.contour(lon_pop, lat_pop, sati.mean(0), [0],
                    colors='k', linewidths=1.4, linestyles='solid', alpha=0.6)
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
ax.text(0.03, 0.88, 'e)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)

ax = axs[0,2]
li = 4
# ax.set_ylabel('AVISO')
tit_fig = 'JJA'
fi = np.where((month==6) | (month==7) | (month==8))
sati = ssh_sat[fi].mean(0) - sshavg_sat
usati = u_sat[fi].mean(0) - uavg_sat
vsati = v_sat[fi].mean(0) - vavg_sat
usati = np.ma.masked_greater(usati, 50)
vsati = np.ma.masked_greater(vsati, 50)
usati = np.ma.masked_less(usati, -50)
vsati = np.ma.masked_less(vsati, -50)
ctop = ax.contourf(lon_sat, lat_sat, sati, levels_sat,
                    extend='both', cmap=colori)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=1)
ax.contour(lon_sat, lat_sat, sati, levels=np.arange(-10,12,2),
                    colors='k', linewidths=1, alpha=0.5)
ax.contour(lon_sat, lat_sat, sati, [0],
                    colors='k', linewidths=1.4, linestyles='solid', alpha=0.6)
for c in lcs.collections:
  c.set_linestyle('solid')
ax.set_title(tit_fig)
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
ax.text(0.03, 0.88, 'c)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)
ax.text(0.12, 1.17, 'southeast monsoon',
        transform=ax.transAxes, color='k', fontsize=16)

ax = axs[1,2]
li = 10
tit_fig = 'POP'
fi = np.where((month_pop==6) | (month_pop==7) | (month_pop==8))
sati = ssh_pop[fi].mean(0) - ssh_pop
usati = u_pop[fi].mean(0) - uavg_pop
vsati = v_pop[fi].mean(0) - vavg_pop
usati = np.ma.masked_greater(usati, 50)
vsati = np.ma.masked_greater(vsati, 50)
usati = np.ma.masked_less(usati, -50)
vsati = np.ma.masked_less(vsati, -50)
ctop = ax.contourf(lon_pop, lat_pop, sati.mean(0), levels_pop,
                    extend='both', cmap=colori)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=1)
ax.contour(lon_pop, lat_pop, sati.mean(0), levels=np.arange(-10,12,2),
                    colors='k', linewidths=1, alpha=0.5)
ax.contour(lon_pop, lat_pop, sati.mean(0), [0],
                    colors='k', linewidths=1.4, linestyles='solid', alpha=0.6)
for c in lcs.collections:
  c.set_linestyle('solid')
# ax.set_title(tit_fig)
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
ax.text(0.03, 0.88, 'g)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)


ax = axs[0,1]
li = 4
# ax.set_ylabel('AVISO')
tit_fig = 'MAM'
fi = np.where((month==3) | (month==4) | (month==5))
sati = ssh_sat[fi].mean(0) - sshavg_sat
usati = u_sat[fi].mean(0) - uavg_sat
vsati = v_sat[fi].mean(0) - vavg_sat
usati = np.ma.masked_greater(usati, 50)
vsati = np.ma.masked_greater(vsati, 50)
usati = np.ma.masked_less(usati, -50)
vsati = np.ma.masked_less(vsati, -50)
ctop = ax.contourf(lon_sat, lat_sat, sati, levels_sat,
                    extend='both', cmap=colori)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=1)
ax.contour(lon_sat, lat_sat, sati, levels=np.arange(-10,12,2),
                    colors='k', linewidths=1, alpha=0.5)
ax.contour(lon_sat, lat_sat, sati, [0],
                    colors='k', linewidths=1.4, linestyles='solid', alpha=0.6)
for c in lcs.collections:
  c.set_linestyle('solid')
ax.set_title(tit_fig)
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
ax.text(0.03, 0.88, 'b)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)


ax = axs[1,1]
li = 10
tit_fig = 'POP'
fi = np.where((month_pop==3) | (month_pop==4) | (month_pop==5))
sati = ssh_pop[fi].mean(0) - ssh_pop
usati = u_pop[fi].mean(0) - uavg_pop
vsati = v_pop[fi].mean(0) - vavg_pop
usati = np.ma.masked_greater(usati, 50)
vsati = np.ma.masked_greater(vsati, 50)
usati = np.ma.masked_less(usati, -50)
vsati = np.ma.masked_less(vsati, -50)
ctop = ax.contourf(lon_pop, lat_pop, sati.mean(0), levels_pop,
                    extend='both', cmap=colori)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=1)
ax.contour(lon_pop, lat_pop, sati.mean(0), levels=np.arange(-10,12,2),
                    colors='k', linewidths=1, alpha=0.5)
ax.contour(lon_pop, lat_pop, sati.mean(0), [0],
                    colors='k', linewidths=1.4, linestyles='solid', alpha=0.6)
for c in lcs.collections:
  c.set_linestyle('solid')
# ax.set_title(tit_fig)
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
ax.text(0.03, 0.88, 'f)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)


ax = axs[0,3]
li = 4
#ax.set_ylabel('AVISO')
tit_fig = 'SON'
fi = np.where((month==9) | (month==10) | (month==11))
sati = ssh_sat[fi].mean(0) - sshavg_sat
usati = u_sat[fi].mean(0) - uavg_sat
vsati = v_sat[fi].mean(0) - vavg_sat
usati = np.ma.masked_greater(usati, 50)
vsati = np.ma.masked_greater(vsati, 50)
usati = np.ma.masked_less(usati, -50)
vsati = np.ma.masked_less(vsati, -50)
ctop = ax.contourf(lon_sat, lat_sat, sati, levels_sat,
                    extend='both', cmap=colori)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=1)
ax.contour(lon_sat, lat_sat, sati, levels=np.arange(-10,12,2),
                    colors='k', linewidths=1, alpha=0.5)
ax.contour(lon_sat, lat_sat, sati, [0],
                    colors='k', linewidths=1.4, linestyles='solid', alpha=0.6)
for c in lcs.collections:
  c.set_linestyle('solid')
ax.set_title(tit_fig)
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
ax.text(0.03, 0.88, 'd)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)


ax = axs[1,3]
li = 10
tit_fig = 'POP'
fi = np.where((month_pop==9) | (month_pop==10) | (month_pop==11))
sati = ssh_pop[fi].mean(0) - ssh_pop
usati = u_pop[fi].mean(0) - uavg_pop
vsati = v_pop[fi].mean(0) - vavg_pop
usati = np.ma.masked_greater(usati, 50)
vsati = np.ma.masked_greater(vsati, 50)
usati = np.ma.masked_less(usati, -50)
vsati = np.ma.masked_less(vsati, -50)
ctop = ax.contourf(lon_pop, lat_pop, sati.mean(0), levels_pop,
                    extend='both', cmap=colori)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=1)
ax.contour(lon_pop, lat_pop, sati.mean(0), levels=np.arange(-10,12,2),
                    colors='k', linewidths=1, alpha=0.5)
ax.contour(lon_pop, lat_pop, sati.mean(0), [0],
                    colors='k', linewidths=1.4, linestyles='solid', alpha=0.6)
for c in lcs.collections:
  c.set_linestyle('solid')
# ax.set_title(tit_fig)
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
plt.subplots_adjust(hspace=1e-4)
ax.set_xticks(np.arange(40,80,10))
ax.set_xticklabels((r'$40^o$E', r'$50^o$E', r'$60^o$E', r'$70^o$E'))
ax.set_yticks(np.arange(-20,10,10))
ax.set_yticklabels((r'$20^o$S', r'$10^o$S', r'$0^o$S'))
ax.text(0.03, 0.88, 'h)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)

# plt.axis([42, 65, -18, -2])
plt.axis([42, 65, -18, 3])
fig.subplots_adjust(left=0.07, right=0.9, hspace=0.03, wspace=0.01)
cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
cbar = fig.colorbar(ctop, cax=cbar_ax, extend='both', ticks=np.arange(-10,12,2))
cbar.set_label('[cm]')
qk = plt.quiverkey(Q, 0.90, 0.08, 10, r'$10 \frac{cm}{s}$', labelpos='E',
                   coordinates='figure')
plt.savefig('aviso_pop_comp_seas_large_contour.pdf', bbox_inches='tight')


llevels = np.arange(0, 5000, 1000)
fig, axs  = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(15.75, 7.22))
ax = axs[0,0]
ax.set_ylabel('AVISO')
tit_fig = 'September'
fi = np.where((month==9))
sati = ssh_sat[fi].mean(0) - sshavg_sat
usati = u_sat[fi].mean(0) - uavg_sat
vsati = v_sat[fi].mean(0) - vavg_sat
usati = np.ma.masked_greater(usati, 50)
vsati = np.ma.masked_greater(vsati, 50)
usati = np.ma.masked_less(usati, -50)
vsati = np.ma.masked_less(vsati, -50)
ctop = ax.contourf(lon_sat, lat_sat, sati, levels_sat,
                    extend='both', cmap=colori)
ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')
ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=0.5)
ax.contour(lon_sat, lat_sat, sati, levels=np.arange(-10,12,2),
                    colors='k', linewidths=1, alpha=0.5)
#Q = ax.quiver(lon_sat[::li, ::li], lat_sat[::li, ::li], usati[::li, ::li],
#        vsati[::li, ::li],
#              scale=150, headlength=15, headwidth=10)
ax.set_title(tit_fig)
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
ax.text(0.03, 0.88, 'a)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)
ax.text(0.12, 1.17, 'northwest monsoon',
        transform=ax.transAxes, color='k', fontsize=16)

ax = axs[1,0]
li = 10
tit_fig = 'POP'
ax.set_ylabel('POP')
fi = np.where((month_pop==9))
sati = ssh_pop[fi].mean(0) - ssh_pop
usati = u_pop[fi].mean(0) - uavg_pop
vsati = v_pop[fi].mean(0) - vavg_pop
usati = np.ma.masked_greater(usati, 50)
vsati = np.ma.masked_greater(vsati, 50)
usati = np.ma.masked_less(usati, -50)
vsati = np.ma.masked_less(vsati, -50)
ctop = ax.contourf(lon_pop, lat_pop, sati.mean(0), levels_pop,
                    extend='both', cmap=colori)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=1)
ax.contour(lon_pop, lat_pop, sati.mean(0), levels=np.arange(-10,12,2),
                    colors='k', linewidths=1, alpha=0.5)
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
ax.text(0.03, 0.88, 'e)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)

ax = axs[0,2]
li = 4
# ax.set_ylabel('AVISO')
tit_fig = 'November'
fi = np.where((month==11))
sati = ssh_sat[fi].mean(0) - sshavg_sat
usati = u_sat[fi].mean(0) - uavg_sat
vsati = v_sat[fi].mean(0) - vavg_sat
usati = np.ma.masked_greater(usati, 50)
vsati = np.ma.masked_greater(vsati, 50)
usati = np.ma.masked_less(usati, -50)
vsati = np.ma.masked_less(vsati, -50)
ctop = ax.contourf(lon_sat, lat_sat, sati, levels_sat,
                    extend='both', cmap=colori)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=1)
ax.contour(lon_sat, lat_sat, sati, levels=np.arange(-10,12,2),
                    colors='k', linewidths=1, alpha=0.5)
for c in lcs.collections:
  c.set_linestyle('solid')
ax.set_title(tit_fig)
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
ax.text(0.03, 0.88, 'c)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)
ax.text(0.12, 1.17, 'southeast monsoon',
        transform=ax.transAxes, color='k', fontsize=16)

ax = axs[1,2]
li = 10
tit_fig = 'POP'
fi = np.where((month_pop==11))
sati = ssh_pop[fi].mean(0) - ssh_pop
usati = u_pop[fi].mean(0) - uavg_pop
vsati = v_pop[fi].mean(0) - vavg_pop
usati = np.ma.masked_greater(usati, 50)
vsati = np.ma.masked_greater(vsati, 50)
usati = np.ma.masked_less(usati, -50)
vsati = np.ma.masked_less(vsati, -50)
ctop = ax.contourf(lon_pop, lat_pop, sati.mean(0), levels_pop,
                    extend='both', cmap=colori)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=1)
ax.contour(lon_pop, lat_pop, sati.mean(0), levels=np.arange(-10,12,2),
                    colors='k', linewidths=1, alpha=0.5)
for c in lcs.collections:
  c.set_linestyle('solid')
# ax.set_title(tit_fig)
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
ax.text(0.03, 0.88, 'g)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)


ax = axs[0,1]
li = 4
# ax.set_ylabel('AVISO')
tit_fig = 'October'
fi = np.where((month==10))
sati = ssh_sat[fi].mean(0) - sshavg_sat
usati = u_sat[fi].mean(0) - uavg_sat
vsati = v_sat[fi].mean(0) - vavg_sat
usati = np.ma.masked_greater(usati, 50)
vsati = np.ma.masked_greater(vsati, 50)
usati = np.ma.masked_less(usati, -50)
vsati = np.ma.masked_less(vsati, -50)
ctop = ax.contourf(lon_sat, lat_sat, sati, levels_sat,
                    extend='both', cmap=colori)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=1)
ax.contour(lon_sat, lat_sat, sati, levels=np.arange(-10,12,2),
                    colors='k', linewidths=1, alpha=0.5)
for c in lcs.collections:
  c.set_linestyle('solid')
ax.set_title(tit_fig)
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
ax.text(0.03, 0.88, 'b)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)


ax = axs[1,1]
li = 10
tit_fig = 'POP'
fi = np.where((month_pop==10))
sati = ssh_pop[fi].mean(0) - ssh_pop
usati = u_pop[fi].mean(0) - uavg_pop
vsati = v_pop[fi].mean(0) - vavg_pop
usati = np.ma.masked_greater(usati, 50)
vsati = np.ma.masked_greater(vsati, 50)
usati = np.ma.masked_less(usati, -50)
vsati = np.ma.masked_less(vsati, -50)
ctop = ax.contourf(lon_pop, lat_pop, sati.mean(0), levels_pop,
                    extend='both', cmap=colori)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=1)
ax.contour(lon_pop, lat_pop, sati.mean(0), levels=np.arange(-10,12,2),
                    colors='k', linewidths=1, alpha=0.5)
for c in lcs.collections:
  c.set_linestyle('solid')
# ax.set_title(tit_fig)
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
ax.text(0.03, 0.88, 'f)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)


ax = axs[0,3]
li = 4
#ax.set_ylabel('AVISO')
tit_fig = 'December'
fi = np.where((month==12))
sati = ssh_sat[fi].mean(0) - sshavg_sat
usati = u_sat[fi].mean(0) - uavg_sat
vsati = v_sat[fi].mean(0) - vavg_sat
usati = np.ma.masked_greater(usati, 50)
vsati = np.ma.masked_greater(vsati, 50)
usati = np.ma.masked_less(usati, -50)
vsati = np.ma.masked_less(vsati, -50)
ctop = ax.contourf(lon_sat, lat_sat, sati, levels_sat,
                    extend='both', cmap=colori)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=1)
ax.contour(lon_sat, lat_sat, sati, levels=np.arange(-10,12,2),
                    colors='k', linewidths=1, alpha=0.5)
for c in lcs.collections:
  c.set_linestyle('solid')
ax.set_title(tit_fig)
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
ax.text(0.03, 0.88, 'd)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)


ax = axs[1,3]
li = 10
tit_fig = 'POP'
fi = np.where((month_pop==12))
sati = ssh_pop[fi].mean(0) - ssh_pop
usati = u_pop[fi].mean(0) - uavg_pop
vsati = v_pop[fi].mean(0) - vavg_pop
usati = np.ma.masked_greater(usati, 50)
vsati = np.ma.masked_greater(vsati, 50)
usati = np.ma.masked_less(usati, -50)
vsati = np.ma.masked_less(vsati, -50)
ctop = ax.contourf(lon_pop, lat_pop, sati.mean(0), levels_pop,
                    extend='both', cmap=colori)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='#d2b466')
lcs = ax.contour(lons, lats, etopo * -1, [0, 200, 1000], colors='#606060', linewidths=1)
ax.contour(lon_pop, lat_pop, sati.mean(0), levels=np.arange(-10,12,2),
                    colors='k', linewidths=1, alpha=0.5)
for c in lcs.collections:
  c.set_linestyle('solid')
# ax.set_title(tit_fig)
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
ax.set_rasterized(True)
plt.subplots_adjust(hspace=1e-4)
ax.set_xticks(np.arange(40,80,10))
ax.set_xticklabels((r'$40^o$E', r'$50^o$E', r'$60^o$E', r'$70^o$E'))
ax.set_yticks(np.arange(-20,10,10))
ax.set_yticklabels((r'$20^o$S', r'$10^o$S', r'$0^o$S'))
ax.text(0.03, 0.88, 'h)',
        transform=ax.transAxes, bbox=dict(facecolor='white',
         alpha=0.5), color='k', fontsize=14)

# plt.axis([42, 65, -18, -2])
plt.axis([42, 65, -18, 3])

fig.subplots_adjust(left=0.07, right=0.9, hspace=0.03, wspace=0.01)
cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
cbar = fig.colorbar(ctop, cax=cbar_ax, extend='both', ticks=np.arange(-10,12,2))
cbar.set_label('[cm]')
qk = plt.quiverkey(Q, 0.90, 0.08, 10, r'$10 \frac{cm}{s}$', labelpos='E',
                   coordinates='figure')
plt.savefig('aviso_pop_comp_seas_large_contourmonth.pdf', bbox_inches='tight')



colori = cm.afmhot_r
llevels = np.arange(0, 5000, 1000)
fig, axs  = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(15.75, 7.22))
ax = axs[0,0]
tit_fig = 'DJF'
fi = np.where((month==12) | (month==1) | (month==2))
sati = ssh_sat[fi].var(0)
ctop = ax.contourf(lon_sat, lat_sat, sati, levels=np.arange(0,85,5),
                    extend='both', cmap=colori)
lcs = ax.contour(lons, lats, etopo, [-1000, -200, -100, 0], colors='#606060', linewidths=1)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='k', alpha=0.5)
for c in lcs.collections:
  c.set_linestyle('solid')
ax.set_title('AVISO')
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
ax.set_ylabel('AVISO')
ax.set_title(tit_fig)

ax = axs[1,0]
li = 4
ax.set_ylabel('POP')
fi = np.where((month_pop==12) | (month_pop==1) | (month_pop==2))
sati = ssh_pop[fi].var(0)
ctop = ax.contourf(lon_pop, lat_pop, sati, levels=np.arange(0,85,5),
                    extend='both', cmap=colori)
lcs = ax.contour(lons, lats, etopo, [-1000, -200, -100, 0], colors='#606060', linewidths=1)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='k', alpha=0.5)
for c in lcs.collections:
  c.set_linestyle('solid')
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)

ax = axs[0,1]
tit_fig = 'MAM'
fi = np.where((month==3) | (month==4) | (month==5))
sati = ssh_sat[fi].var(0)
ctop = ax.contourf(lon_sat, lat_sat, sati, levels=np.arange(0,85,5),
                    extend='both', cmap=colori)
lcs = ax.contour(lons, lats, etopo, [-1000, -200, -100, 0], colors='#606060', linewidths=1)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='k', alpha=0.5)
for c in lcs.collections:
  c.set_linestyle('solid')
ax.set_title('AVISO')
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
ax.set_title(tit_fig)

ax = axs[1,1]
li = 4
fi = np.where((month_pop==3) | (month_pop==4) | (month_pop==5))
sati = ssh_pop[fi].var(0)
ctop = ax.contourf(lon_pop, lat_pop, sati, levels=np.arange(0,85,5),
                    extend='both', cmap=colori)
lcs = ax.contour(lons, lats, etopo, [-1000, -200, -100, 0], colors='#606060', linewidths=1)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='k', alpha=0.5)
for c in lcs.collections:
  c.set_linestyle('solid')
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)

ax = axs[0,2]
tit_fig = 'JJA'
fi = np.where((month==6) | (month==7) | (month==8))
sati = ssh_sat[fi].var(0)
ctop = ax.contourf(lon_sat, lat_sat, sati, levels=np.arange(0,85,5),
                    extend='both', cmap=colori)
lcs = ax.contour(lons, lats, etopo, [-1000, -200, -100, 0], colors='#606060', linewidths=1)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='k', alpha=0.5)
for c in lcs.collections:
  c.set_linestyle('solid')
ax.set_title('AVISO')
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
ax.set_title(tit_fig)

ax = axs[1,2]
li = 4
fi = np.where((month_pop==6) | (month_pop==7) | (month_pop==8))
sati = ssh_pop[fi].var(0)
ctop = ax.contourf(lon_pop, lat_pop, sati, levels=np.arange(0,85,5),
                    extend='both', cmap=colori)
lcs = ax.contour(lons, lats, etopo, [-1000, -200, -100, 0], colors='#606060', linewidths=1)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='k', alpha=0.5)
for c in lcs.collections:
  c.set_linestyle('solid')
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)


ax = axs[0,3]
tit_fig = 'SON'
fi = np.where((month==9) | (month==10) | (month==11))
sati = ssh_sat[fi].var(0)
ctop = ax.contourf(lon_sat, lat_sat, sati, levels=np.arange(0,85,5),
                    extend='both', cmap=colori)
lcs = ax.contour(lons, lats, etopo, [-1000, -200, -100, 0], colors='#606060', linewidths=1)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='k', alpha=0.5)
for c in lcs.collections:
  c.set_linestyle('solid')
ax.set_title('AVISO')
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
ax.set_title(tit_fig)

ax = axs[1,3]
li = 4
fi = np.where((month_pop==9) | (month_pop==10) | (month_pop==11))
sati = ssh_pop[fi].var(0)
ctop = ax.contourf(lon_pop, lat_pop, sati, levels=np.arange(0,85,5),
                    extend='both', cmap=colori)
lcs = ax.contour(lons, lats, etopo, [-1000, -200, -100, 0], colors='#606060', linewidths=1)
lcs = ax.contourf(lons, lats, etopo, llevels, colors='k', alpha=0.5)
for c in lcs.collections:
  c.set_linestyle('solid')
ax.set(adjustable='box-forced', aspect='equal')
ax.set_rasterized(True)
plt.axis([42, 65, -18, 3])


plt.subplots_adjust(hspace=1e-4)
ax.set_xticks(np.arange(40,80,10))
ax.set_xticklabels((r'$40^o$E', r'$50^o$E', r'$60^o$E', r'$70^o$E'))
ax.set_yticks(np.arange(-20,10,10))
ax.set_yticklabels((r'$20^o$S', r'$10^o$S', r'$0^o$S'))

fig.subplots_adjust(left=0.05, right=0.95, hspace=0.03, wspace=0.01)
cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
cbar = fig.colorbar(ctop, cax=cbar_ax, extend='both')
cbar.set_label('$[cm^2]$')
# plt.savefig('aviso_pop_comp_seas_large_variance.pdf', bbox_inches='tight')
