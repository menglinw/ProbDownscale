import os
import sys
import utils.data_processing as data_processing
import math
import numpy as np
import netCDF4 as nc
from matplotlib import pyplot as plt
from math import exp, sqrt, log
import time
import geopandas as gpd
import pandas as pd

class evaluate_data():
    def __init__(self, save_path):
        self.save_path = save_path
        self.target_var = 'TOTEXTTAU'
        self.lats_lons, self.test_g_data, self.test_m_data = self._load_data()
        self.d_data, self.d_data_meta, self.d_data_meta_beta = self.read_downscale_data()
        self.e_data, self.e_data_meta, self.e_data_meta_beta = self.read_epochs_data()
        self.evaluate_downscale(self.d_data, self.test_g_data, 'downscale_data_R2.jpg', 'R2')
        self.evaluate_downscale(self.d_data_meta, self.test_g_data, 'downscale_data_meta_R2.jpg', 'meta R2')
        self.evaluate_downscale(self.d_data_meta_beta, self.test_g_data, 'downscale_data_meta_beta_R2.jpg', 'meta beta R2')
        self.evaluate_epochs(self.e_data, 'direct_downscale_epochs.jpg')
        self.evaluate_epochs(self.e_data_meta, 'meta_downscale_epochs.jpg')
        self.evaluate_epochs(self.e_data_meta_beta, 'meta_beta_downscale_epochs.jpg')

    def _load_data(self):
        file_path_g_06 = '/project/mereditf_284/menglin/Downscale_data/MERRA2/G5NR_aerosol_variables_over_MiddleEast_daily_20060516-20070515.nc'
        file_path_g_05 = '/project/mereditf_284/menglin/Downscale_data/MERRA2/G5NR_aerosol_variables_over_MiddleEast_daily_20050516-20060515.nc'
        file_path_m = '/project/mereditf_284/menglin/Downscale_data/MERRA2/MERRA2_aerosol_variables_over_MiddleEast_daily_20000516-20180515.nc'
        file_path_country = ['/project/mereditf_284/menglin/Downscale_data/Country_shape/AFG_adm/AFG_adm0.shp',
                             '/project/mereditf_284/menglin/Downscale_data/Country_shape/ARE_adm/ARE_adm0.shp',
                             '/project/mereditf_284/menglin/Downscale_data/Country_shape/IRQ_adm/IRQ_adm0.shp',
                             '/project/mereditf_284/menglin/Downscale_data/Country_shape/KWT_adm/KWT_adm0.shp',
                             '/project/mereditf_284/menglin/Downscale_data/Country_shape/QAT_adm/QAT_adm0.shp',
                             '/project/mereditf_284/menglin/Downscale_data/Country_shape/SAU_adm/SAU_adm0.shp']
        # read data
        g05_data = nc.Dataset(file_path_g_05)
        g06_data = nc.Dataset(file_path_g_06)
        m_data_nc = nc.Dataset(file_path_m)

        # define lat&lon of MERRA, G5NR and mete
        M_lons = m_data_nc.variables['lon'][:]
        # self.M_lons = (M_lons-M_lons.mean())/M_lons.std()
        M_lats = m_data_nc.variables['lat'][:]
        # self.M_lats = (M_lats-M_lats.mean())/M_lats.std()
        G_lons = g05_data.variables['lon'][:]
        # self.G_lons = (G_lons-G_lons.mean())/G_lons.std()
        G_lats = g05_data.variables['lat'][:]

        # extract target data
        g_data = np.concatenate((g05_data.variables[self.target_var], g06_data.variables[self.target_var]), axis=0)
        m_data = m_data_nc.variables[self.target_var][5 * 365:7 * 365, :, :]

        # load country file
        country_shape = gpd.read_file(file_path_country[0])
        for country_path in file_path_country[1:]:
            country_shape = pd.concat([country_shape, gpd.read_file(country_path)])

        # get outer bound
        latmin, lonmin, latmax, lonmax = country_shape.total_bounds
        latmin_ind = np.argmin(np.abs(G_lats - latmin))
        latmax_ind = np.argmin(np.abs(G_lats - latmax))
        lonmin_ind = np.argmin(np.abs(G_lons - lonmin))
        lonmax_ind = np.argmin(np.abs(G_lons - lonmax))
        # 123 * 207
        g_data = g_data[:, latmin_ind - 1:latmax_ind + 1, lonmin_ind:lonmax_ind + 2]

        G_lats = G_lats[latmin_ind - 1:latmax_ind + 1]
        G_lons = G_lons[lonmin_ind:lonmax_ind + 2]

        print('Data shape:', g_data.shape)

        # log and normalize data
        g_data = self._normalize(np.log(g_data))
        m_data = self._normalize(np.log(m_data))

        # split data into traing and test
        train_g_data, test_g_data = g_data[:657], g_data[657:]
        train_m_data, test_m_data = m_data[:657], m_data[657:]
        lats_lons = [G_lats, G_lons, M_lats, M_lons]
        return lats_lons, test_g_data, test_m_data

    def _normalize(self, data):
        return (data - data.min()) / (data.max() - data.min())

    def read_downscale_data(self):
        d_data = np.zeros_like(self.test_g_data)
        d_data_meta = np.zeros_like(self.test_g_data)
        d_data_meta_beta = np.zeros_like(self.test_g_data)
        for lat in list('1234'):
            for lon in list('1234567'):
                length_lat = 30 if lat != '4' else 33
                d_part = np.load(os.path.join(self.save_path, 'downscaled_data_'+lat+lon+'.npy'))
                dm_part = np.load(os.path.join(self.save_path, 'downscaled_data_'+lat+lon+'_meta'+'.npy'))
                dmb_part = np.load(os.path.join(self.save_path, 'downscaled_data_'+lat+lon+'_meta'+'_beta'+'.npy'))
                d_data[:, (int(lat) - 1) * 30:((int(lat) - 1) * 30 + length_lat),
                (int(lon) - 1) * 30:int(lon) * 30] = d_part
                d_data_meta[:, (int(lat) - 1) * 30:((int(lat) - 1) * 30 + length_lat),
                (int(lon) - 1) * 30:int(lon) * 30] = dm_part
                d_data_meta_beta[:, (int(lat) - 1) * 30:((int(lat) - 1) * 30 + length_lat),
                (int(lon) - 1) * 30:int(lon) * 30] = dmb_part
        return d_data, d_data_meta, d_data_meta_beta


    def read_epochs_data(self):
        d_data = np.zeros((2, 0))
        d_data_meta = np.zeros((2, 0))
        d_data_meta_beta = np.zeros((2, 0))
        for lat in list('1234'):
            for lon in list('1234567'):
                d_part = np.load(os.path.join(self.save_path, 'epochs_data_'+lat+lon+'.npy'))
                d_data = np.concatenate([d_data, d_part], axis=1)
                dm_part = np.load(os.path.join(self.save_path, 'epochs_data_'+lat+lon+'_meta'+'.npy'))
                d_data_meta = np.concatenate([d_data_meta, dm_part], axis=1)
                dmb_part = np.load(os.path.join(self.save_path, 'epochs_data_'+lat+lon+'_meta'+'_beta'+'.npy'))
                d_data_meta_beta = np.concatenate([d_data_meta_beta, dmb_part], axis=1)
        return d_data, d_data_meta, d_data_meta_beta

    def read_meta_history(self):
        pass

    def _image_evaluate(self, pred_data, true_data):
        if pred_data.shape != true_data.shape:
            print('Please check data consistency!')
            raise ValueError
        length = np.prod(pred_data.shape[1:])
        r2_list = np.zeros(pred_data.shape[0])
        rmse_list = np.zeros(pred_data.shape[0])
        filter = ~np.isnan(pred_data[0].reshape(length))
        for i in range(pred_data.shape[0]):
            r2_list[i], _ = data_processing.rsquared(pred_data[i].reshape(length)[filter],
                                                     true_data[i].reshape(length)[filter])
            rmse_list[i] = np.nanmean(np.square(pred_data[i] - true_data[i]))
        return rmse_list, r2_list

    def evaluate_downscale(self, d_data, t_data, fig_name, title):
        print('Missing Rate:', np.mean(np.isnan(d_data)))
        rmse_list, r2_list = self._image_evaluate(d_data, t_data)
        plt.figure()
        #plt.plot(rmse_list/100, "-b",label='RMSE')
        plt.plot(r2_list, "-r", label='R2')
        plt.title(title)
        plt.legend()
        plt.show()
        plt.savefig(os.path.join(self.save_path, fig_name))
        plt.close()

        plt.figure()
        plt.hist(d_data.reshape((1, np.prod(d_data.shape))), bins=100, alpha=0.5, label='downscaled')
        plt.hist(t_data.reshape((1, np.prod(t_data.shape))), bins=100, alpha=0.5, label='true')
        plt.legend()
        plt.show()
        plt.savefig(os.path.join(self.save_path, 'distribut_'+fig_name))
        plt.close()

    def evaluate_epochs(self, e_data, fig_name):
        plt.figure()
        plt.hist(e_data[0], bins=100, alpha=0.5, label='prob')
        plt.hist(e_data[1], bins=100, alpha=0.5, label='reg')
        plt.legend()
        plt.show()
        plt.savefig(os.path.join(self.save_path, fig_name))
        plt.close()

if __name__ == '__main__':
    save_path = sys.argv[1]
    evaluate_data(save_path)
