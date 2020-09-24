# import netCDF satellites files as S
import netCDF4 as S
import numpy as np
import cmocean
import sys
sys.path.append('/Users/carolina/SLOMO/codes')
import matplotlib.pyplot as plt
from matplotlib.dates import date2num, num2date, datetime
from matplotlib import rcParams as rcp
from mpl_toolkits.basemap import Basemap
from slomo_processing import *
from matplotlib import cm
import matplotlib
import cmocean
rcp['font.size'] = 12.
rcp['ytick.right'] = True
rcp['ytick.labelright'] = False

# load the bathymetry8mammimi8

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

fileobj = S.Dataset('1993-1996_3590E_-205lat.nc', mode='r')
u0 = fileobj.variables['ugosa'][:].squeeze() # absolute geostrophic velocity, ugosa is the geostrophic velocity anomaly referenced to 1993-2012
v0 = fileobj.variables['vgosa'][:].squeeze()
ssh0 = fileobj.variables['sla'][:] # sea level anomaly
t0 = fileobj.variables['time'][:] + date2num(datetime.datetime.strptime('1950-01-01 00:00:00', '%Y-%d-%m %H:%M:%S')) # divide by 24 if in hours

fileobj = S.Dataset('1997-2000_3590E_-205lat.nc', mode='r')
u1 = fileobj.variables['ugosa'][:].squeeze() # absolute geostrophic velocity, ugosa is the geostrophic velocity anomaly referenced to 1993-2012
v1 = fileobj.variables['vgosa'][:].squeeze()
ssh1 = fileobj.variables['sla'][:] # sea level anomaly
t1 = fileobj.variables['time'][:] + date2num(datetime.datetime.strptime('1950-01-01 00:00:00', '%Y-%d-%m %H:%M:%S')) # divide by 24 if in hours

fileobj = S.Dataset('2001-2004_3590E_-205lat.nc', mode='r')
u2 = fileobj.variables['ugosa'][:].squeeze() # absolute geostrophic velocity, ugosa is the geostrophic velocity anomaly referenced to 1993-2012
v2 = fileobj.variables['vgosa'][:].squeeze()
ssh2 = fileobj.variables['sla'][:] # sea level anomaly
t2 = fileobj.variables['time'][:] + date2num(datetime.datetime.strptime('1950-01-01 00:00:00', '%Y-%d-%m %H:%M:%S')) # divide by 24 if in hours

fileobj = S.Dataset('2005-2008_3590E_-205lat.nc', mode='r')
u3 = fileobj.variables['ugosa'][:].squeeze() # absolute geostrophic velocity, ugosa is the geostrophic velocity anomaly referenced to 1993-2012
v3 = fileobj.variables['vgosa'][:].squeeze()
ssh3 = fileobj.variables['sla'][:] # sea level anomaly
t3 = fileobj.variables['time'][:] + date2num(datetime.datetime.strptime('1950-01-01 00:00:00', '%Y-%d-%m %H:%M:%S')) # divide by 24 if in hours

fileobj = S.Dataset('2009-2012_3590E_-205lat.nc', mode='r')
u4 = fileobj.variables['ugosa'][:].squeeze() # absolute geostrophic velocity, ugosa is the geostrophic velocity anomaly referenced to 1993-2012
v4 = fileobj.variables['vgosa'][:].squeeze()
ssh4 = fileobj.variables['sla'][:] # sea level anomaly
t4 = fileobj.variables['time'][:] + date2num(datetime.datetime.strptime('1950-01-01 00:00:00', '%Y-%d-%m %H:%M:%S')) # divide by 24 if in hours

fileobj = S.Dataset('2013-2016_3590E_-205lat.nc', mode='r')
u5 = fileobj.variables['ugosa'][:].squeeze() # absolute geostrophic velocity, ugosa is the geostrophic velocity anomaly referenced to 1993-2012
v5 = fileobj.variables['vgosa'][:].squeeze()
ssh5 = fileobj.variables['sla'][:] # sea level anomaly
t5 = fileobj.variables['time'][:] + date2num(datetime.datetime.strptime('1950-01-01 00:00:00', '%Y-%d-%m %H:%M:%S')) # divide by 24 if in hours

fileobj = S.Dataset('2017-2018_3590E_-205lat.nc', mode='r')
u6 = fileobj.variables['ugosa'][:].squeeze() # absolute geostrophic velocity, ugosa is the geostrophic velocity anomaly referenced to 1993-2012
v6 = fileobj.variables['vgosa'][:].squeeze()
ssh6 = fileobj.variables['sla'][:] # sea level anomaly
t6 = fileobj.variables['time'][:] + date2num(datetime.datetime.strptime('1950-01-01 00:00:00', '%Y-%d-%m %H:%M:%S')) # divide by 24 if in hours

fileobj = S.Dataset('2019_3590E_-205lat.nc',mode='r')
u7 = fileobj.variables['ugosa'][:].squeeze() # absolute geostrophic velocity, ugosa is the geostrophic velocity anomaly referenced to 1993-2012
v7 = fileobj.variables['vgosa'][:].squeeze()
ssh7 = fileobj.variables['sla'][:] # sea level anomaly
t7 = fileobj.variables['time'][:] + date2num(datetime.datetime.strptime('1950-01-01 00:00:00', '%Y-%d-%m %H:%M:%S')) # divide by 24 if in hours
lat = fileobj.variables['latitude'][:]
lon = fileobj.variables['longitude'][:]

u = np.concatenate((u0, u1, u2, u3, u4, u5, u6, u7), axis=0)
v = np.concatenate((v0, v1, v2, v3, v4, v5, v6, v7), axis=0)
ssh = np.concatenate((ssh0, ssh1, ssh2, ssh3, ssh4, ssh5, ssh6, ssh7), axis=0)
t = np.concatenate((t0, t1, t2, t3, t4, t5, t6, t7), axis=0)


ssh = np.ma.masked_less_equal(ssh, -1e9)
mag = np.sqrt(u ** 2 + v ** 2)
 # divide by 24 if in hours
size = t.size; year = np.zeros(size); month = np.zeros(size)
# choose only specific dates
for ii in np.arange(0, size):
    year[ii] = num2date(t[ii]).year
    month[ii] = num2date(t[ii]).month

lati1 = -4
loni =  55.646
junk, fi_x1 = find_nearest(lat, lati1)
ssh_sat = ssh[:, fi_x1, :]

colori = cmocean.cm.balance
linilow=-25
linihigh=25
levels = np.arange(linilow, linihigh + 1, 1)
fig, axs  = plt.subplots(1, 2, sharex=True, figsize=(9.6, 7.3))
ax = axs[1]
ctop = ax.contourf(lon, num2date(t), ssh_sat * 100, cmap=colori, levels=levels, extend='both')
ax.contour(lon, num2date(t), ssh_sat *100, [0], colors='k', alpha=0.5, linewidths=0.8)
ax.axvline(x=54, color='k',linestyle=':',lw=1, alpha=0.8)
ax.axvline(x=57, color='k',linestyle=':',lw=1, alpha=0.8)
fi = date2num(datetime.datetime.strptime('2017-01-1', '%Y-%m-%d'))
ax.axhline(y=fi, color='k', linestyle='-', lw=1.5, alpha=0.3)
fi = date2num(datetime.datetime.strptime('2018-01-1', '%Y-%m-%d'))
ax.axhline(y=fi, color='k', linestyle='-', lw=1.5, alpha=0.3)
fi1 = date2num(datetime.datetime.strptime('2016-01', '%Y-%m'))
fi2 = date2num(datetime.datetime.strptime('2019-01', '%Y-%m'))
plt.axis([40, 90, fi1, fi2])
f1 = date2num(datetime.datetime.strptime('2016-03-01', '%Y-%m-%d'))
f2 = date2num(datetime.datetime.strptime('2016-06-01', '%Y-%m-%d'))
f3 = date2num(datetime.datetime.strptime('2016-09-01', '%Y-%m-%d'))
f4 = date2num(datetime.datetime.strptime('2016-12-01', '%Y-%m-%d'))
f5 = date2num(datetime.datetime.strptime('2017-03-01', '%Y-%m-%d'))
f6 = date2num(datetime.datetime.strptime('2017-06-01', '%Y-%m-%d'))
f7 = date2num(datetime.datetime.strptime('2017-09-01', '%Y-%m-%d'))
f8 = date2num(datetime.datetime.strptime('2017-12-01', '%Y-%m-%d'))
f9 = date2num(datetime.datetime.strptime('2018-03-01', '%Y-%m-%d'))
f10 = date2num(datetime.datetime.strptime('2018-06-01', '%Y-%m-%d'))
f11 = date2num(datetime.datetime.strptime('2018-09-01', '%Y-%m-%d'))
f12 = date2num(datetime.datetime.strptime('2018-12-01', '%Y-%m-%d'))
ax.set_yticks([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12])
ax.set_yticklabels(('Mar','Jul','Sep','Dec', 'Mar','Jul','Sep','Dec', 'Mar','Jul','Sep', 'Dec'))
cbar_ax = fig.add_axes([0.09, 0.02, 0.84, 0.02])
cbar = fig.colorbar(ctop, extend='both', cax=cbar_ax, ax=ax, orientation='horizontal')
cbar.set_label('SSH [cm]')
ax.set_xticks(np.arange(40,100,10))
ax.set_xticklabels((r'$40^o$E', r'$50^o$E', r'$60^o$E', r'$70^o$E',r'$80^o$E',r'$90^o$E'))
ax.text(32, 736162.0983241181, '2016', fontsize=10,rotation=90)
ax.text(32, 736533.209720972, '2017', fontsize=10, rotation=90)
ax.text(32, 736883.1818610432, '2018', fontsize=10, rotation=90)
ax.text(54.1, 737065, 'SP', fontsize=12)
ax.set_rasterized(True)
# plt.savefig('figures/hov_4S6S_interannual.pdf', bbox_inches='tight')
ax.text(0.03, 1.01, 'b)',
        transform=ax.transAxes, fontsize=16)

# choose time
idx1 = np.where((month == 1) & (year == 1993))[0]
idx2 = np.where((month == 12) & (year == 2009))[-1] # data ends in 2018/06/10
ssh = ssh[idx1[0]:idx2[-1]+1, :, :]
u = u[idx1[0]:idx2[-1]+1, :, :]
v = v[idx1[0]:idx2[-1]+1, :, :]
u = np.ma.masked_less_equal(u, -1e3)
v = np.ma.masked_less_equal(v, -1e3)
ssh = np.ma.masked_less_equal(ssh, -1e9)
t = t[idx1[0]:idx2[-1]+1]

lat = fileobj.variables['latitude'][:]
lon = fileobj.variables['longitude'][:]

size = t.size; year = np.zeros(size); month = np.zeros(size); day = np.zeros(size);
# choose only specific dates
for ii in np.arange(0, size):
    year[ii] = num2date(t[ii]).year
    month[ii] = num2date(t[ii]).month
    day[ii] = num2date(t[ii]).day

lati1 = -4
loni =  55.646
junk, fi_x1 = find_nearest(lat, lati1)
ssh_sat = ssh[:, fi_x1, :]

# delete leap years from 1993 to 2017
fi = date2num(datetime.datetime.strptime('1996-02-02', '%Y-%d-%m'))
ssh_sat = np.delete(ssh_sat, np.where(t == fi)[0][0], 0)
fi = date2num(datetime.datetime.strptime('2000-02-02', '%Y-%d-%m'))
ssh_sat = np.delete(ssh_sat, np.where(t == fi)[0][0], 0)
fi = date2num(datetime.datetime.strptime('2004-02-02', '%Y-%d-%m'))
ssh_sat = np.delete(ssh_sat, np.where(t == fi)[0][0], 0)
fi = date2num(datetime.datetime.strptime('2008-02-02', '%Y-%d-%m'))
ssh_sat = np.delete(ssh_sat, np.where(t == fi)[0][0], 0)
#fi = date2num(datetime.datetime.strptime('2016-02-02', '%Y-%d-%m'))
#ssh_sat = np.delete(ssh_sat, np.where(t == fi)[0][0], 0)
#fi = date2num(datetime.datetime.strptime('2012-02-02', '%Y-%d-%m'))
#ssh_sat = np.delete(ssh_sat, np.where(t == fi)[0][0], 0)
colori = cmocean.cm.balance
#linilow=-15
#linihigh=15
#levels = np.arange(linilow, linihigh + 1, 1)
ssh_syear = np.reshape(ssh_sat, (17, 365, 221))
test2 = ssh_syear.mean(0)
tyear = np.arange(1,365+32,1)
test = np.zeros((365 + 31, 221))
test[:31] = test2[-31:]
test[31:] = test2
ax = axs[0]
# fig, ax  = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(4.71, 6.06))
tit_fig = 'AVISO'
ctop = ax.contourf(lon, tyear, test * 100, cmap=colori, levels=levels, extend='both')
ax.contour(lon, tyear, test*100, [0], colors='k', alpha=0.5, linewidths=0.8)
ax.set_rasterized(True)
ax.axvline(x=54, color='k',linestyle=':',lw=1, alpha=0.8)
ax.axvline(x=57, color='k',linestyle=':',lw=1, alpha=0.8)
ax.axhline(y=91, color='k', linestyle='--', lw=1, alpha=0.3)
ax.axhline(y=182, color='k', linestyle='--', lw=1, alpha=0.3)
ax.axhline(y=274, color='k', linestyle='--', lw=1, alpha=0.3)
ax.set_yticks([0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 365])
ax.set_yticklabels(('Dec','Jan', 'Feb', 'Mar', 'Apr','May', 'Jun','Jul', 'Aug', 'Sep','Oct',
  'Nov','Dec'))
plt.axis([40, 90, 0, 365])
ax.set_xticks(np.arange(40,100,10))
ax.set_xticklabels((r'$40^o$E', r'$50^o$E', r'$60^o$E', r'$70^o$E',r'$80^o$E',r'$90^o$E'))
ax.text(30.5, 232, 'southeast', fontsize=10, rotation=90)
ax.text(32, 230, 'monsoon', fontsize=10,rotation=90)
ax.text(30.5, 47, 'northwest', fontsize=10,rotation=90)
ax.text(32, 45, 'monsoon', fontsize=10, rotation=90)
ax.text(54.1, 398, 'SP', fontsize=12)
ax.text(0.03, 1.01, 'a)',
        transform=ax.transAxes, fontsize=16)
ax.set_rasterized(True)
cbar_ax = fig.add_axes([0.09, 0.02, 0.84, 0.02])
cbar = fig.colorbar(ctop, extend='both', cax=cbar_ax, ax=ax, orientation='horizontal')
cbar.set_label('SSH [cm]')
plt.savefig('/Users/carolina/Dropbox/Seychelles_Paper_Alma/figure_12.pdf', bbox_inches='tight')
