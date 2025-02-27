import pandas as pd
import numpy as np
import os
import openmatrix as omx
import itertools
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

class HighwayNetwork:
    """ This is the class for all information and utilities relating to highway network.
    Holds all highway network related data for a completed ABM scenario model run.
    This includes input highway network data from the report/hwyTcad.csv and loaded highway
    network data from the report/hwyload_<<TOD>>.csv files. Class also
    includes functions to calculate vehicle hours delayed (vhd), vehicle hours
    time (vht), and vehicle miles traveled (vmt).
    
    Args:
        scenario_path: String location of the completed ABM scenario folder
        metric: ['vht','vhd','vmt']
        group: ['timePeriod','FC']
    
    
    """
    def read_hwynetwork(scenario_path)-> pd.DataFrame:
            # create list of loaded highway network files
            # and their respective ABM 5 time of day time periods
            time_periods = ['EA', 'AM', 'MD', 'PM', 'EV']

            # initialize loaded highway network DataFrame
            hwy_load = pd.DataFrame()

            # for each loaded highway network file
            for tp in time_periods:
                # read the loaded highway network
                data = pd.read_csv(
                    os.path.join(scenario_path, 'report','hwyload_' + tp + '.csv'),
                    usecols=["ID1",  # hwycov id
                            "AB_Time",  # A-B direction loaded time
                            "BA_Time",  # B-A direction loaded time
                            "AB_Flow_SOV_NTPL",
                            "BA_Flow_SOV_NTPL",
                            "AB_Flow_SOV_TPL",
                            "BA_Flow_SOV_TPL",
                            "AB_Flow_SR2L",
                            "BA_Flow_SR2L",
                            "AB_Flow_SR3L",
                            "BA_Flow_SR3L",
                            "AB_Flow_SOV_NTPM",
                            "BA_Flow_SOV_NTPM",
                            "AB_Flow_SOV_TPM",
                            "BA_Flow_SOV_TPM",
                            "AB_Flow_SR2M",
                            "BA_Flow_SR2M",
                            "AB_Flow_SR3M",
                            "BA_Flow_SR3M",
                            "AB_Flow_SOV_NTPH",
                            "BA_Flow_SOV_NTPH",
                            "AB_Flow_SOV_TPH",
                            "BA_Flow_SOV_TPH",
                            "AB_Flow_SR2H",
                            "BA_Flow_SR2H",
                            "AB_Flow_SR3H",
                            "BA_Flow_SR3H",
                            "AB_Flow_lhd",
                            "BA_Flow_lhd",
                            "AB_Flow_mhd",
                            "BA_Flow_mhd",
                            "AB_Flow_hhd",
                            "BA_Flow_hhd",
                            "AB_Flow",  # A-B directional flow
                            "BA_Flow"   # B-A directional flow
                            ]
                        )
                data["AB_Flow_Auto"] = data[["AB_Flow_SOV_NTPL",
                                         "AB_Flow_SOV_TPL",
                                         "AB_Flow_SR2L",
                                         "AB_Flow_SR3L",
                                         "AB_Flow_SOV_NTPM",
                                         "AB_Flow_SOV_TPM",
                                         "AB_Flow_SR2M",
                                         "AB_Flow_SR3M",
                                         "AB_Flow_SOV_NTPH",
                                         "AB_Flow_SOV_TPH",
                                         "AB_Flow_SR2H",
                                         "AB_Flow_SR3H"]].sum(axis=1)

                data["BA_Flow_Auto"] = data[["BA_Flow_SOV_NTPL",
                                         "BA_Flow_SOV_TPL",
                                         "BA_Flow_SR2L",
                                         "BA_Flow_SR3L",
                                         "BA_Flow_SOV_NTPM",
                                         "BA_Flow_SOV_TPM",
                                         "BA_Flow_SR2M",
                                         "BA_Flow_SR3M",
                                         "BA_Flow_SOV_NTPH",
                                         "BA_Flow_SOV_TPH",
                                         "BA_Flow_SR2H",
                                         "BA_Flow_SR3H"]].sum(axis=1)

                data["AB_Flow_Truck"] = data[["AB_Flow_lhd",
                                          "AB_Flow_mhd",
                                          "AB_Flow_hhd"]].sum(axis=1)

                data["BA_Flow_Truck"] = data[["BA_Flow_lhd",
                                          "BA_Flow_mhd",
                                          "BA_Flow_hhd"]].sum(axis=1)

                # re-calculate total flow columns
                # issue currently exists in files where flows by mode
                # do not equal the total flows
                data["AB_Flow"] = data[["AB_Flow_Auto",
                                        "AB_Flow_Truck"]].sum(axis=1)

                data["BA_Flow"] = data[["BA_Flow_Auto",
                                        "BA_Flow_Truck"]].sum(axis=1)
                # fill NaN values with 0s
                data.fillna(value=0, inplace=True)                
                
                data['timePeriod'] = tp

                # add to result DataFrame
                hwy_load = hwy_load.append(data)

            hwy_tcad = pd.read_csv(
                os.path.join(scenario_path, 'report','hwyTcad.csv'),
                usecols=["ID",  # hwycov id
                        "Length",  # length of link segment (miles)
                        "FC",  # link functional class
                        "ABTM_EA",  # A-B direction Early AM free flow time (minutes)
                        "ABTM_AM",  # A-B direction AM Peak free flow time (minutes)
                        "ABTM_MD",  # A-B direction Mid-day free flow time (minutes)
                        "ABTM_PM",  # A-B direction PM Peak free flow time (minutes)
                        "ABTM_EV",  # A-B direction Evening free flow time (minutes)
                        "BATM_EA",  # B-A direction Early AM free flow time (minutes)
                        "BATM_AM",  # B-A direction AM Peak free flow time (minutes)
                        "BATM_MD",  # B-A direction Mid-day free flow time (minutes)
                        "BATM_PM",  # B-A direction PM Peak free flow time (minutes)
                        "BATM_EV",  # B-A direction Evening free flow time (minutes)
                        "ABTX_EA",  # A-B direction Early AM intersection delay time (minutes)
                        "ABTX_AM",  # A-B direction AM Peak intersection delay time (minutes)
                        "ABTX_MD",  # A-B direction Mid-day intersection delay time (minutes)
                        "ABTX_PM",  # A-B direction PM Peak intersection delay time (minutes)
                        "ABTX_EV",  # A-B direction Evening intersection delay time (minutes)
                        "BATX_EA",  # B-A direction Early AM intersection delay time (minutes)
                        "BATX_AM",  # B-A direction AM Peak intersection delay time (minutes)
                        "BATX_MD",  # B-A direction Mid-day intersection delay time (minutes)
                        "BATX_PM",  # B-A direction PM Peak intersection delay time (minutes)
                        "BATX_EV",  # B-A direction Evening intersection delay time (minutes)
                        "ABPRELOAD_EA",  # A-B direction Early AM bus volume
                        "ABPRELOAD_AM",  # A-B direction AM Peak bus volume
                        "ABPRELOAD_MD",  # A-B direction Mid-day bus volume
                        "ABPRELOAD_PM",  # A-B direction PM Peak bus volume
                        "ABPRELOAD_EV",  # A-B direction Evening bus volume
                        "BAPRELOAD_EA",  # B-A direction Early AM bus volume
                        "BAPRELOAD_AM",  # B-A direction AM Peak bus volume
                        "BAPRELOAD_MD",  # B-A direction Mid-day bus volume
                        "BAPRELOAD_PM",  # B-A direction PM Peak bus volume
                        "BAPRELOAD_EV"  # B-A direction Evening bus volume]                        
                ]
            )
            # restructure file to long-format by ABM 5 time of day period
            hwy_tcad = pd.wide_to_long(
                df=hwy_tcad,
                stubnames=["ABTM",
                         "BATM",
                         "ABTX",
                         "BATX",
                         "ABPRELOAD",
                         "BAPRELOAD"],
                i=["ID",
                   "Length",
                   "FC"],
                j="timePeriod",
                sep="_",
                suffix="\w+").reset_index()
            
            # fill NaN values with 0s
            hwy_tcad.fillna(value=0, inplace=True)
            # add description of functional class to DataFrame
            conditions = [(hwy_tcad["FC"] == 1),
                          (hwy_tcad["FC"] == 2),
                          (hwy_tcad["FC"] == 3),
                          (hwy_tcad["FC"] == 4),
                          (hwy_tcad["FC"] == 5),
                          (hwy_tcad["FC"] == 6),
                          (hwy_tcad["FC"] == 7),
                          (hwy_tcad["FC"] == 8),
                          (hwy_tcad["FC"] == 9),
                          (hwy_tcad["FC"] == 10)]

            choices = ["Freeway",
                       "Prime Arterial",
                       "Major Arterial",
                       "Collector",
                       "Local Collector",
                       "Rural Collector",
                       "Local Road",
                       "Freeway Connector Ramp",
                       "Local Ramp",
                       "Zone Connector"]

            hwy_tcad["fc_desc"] = np.select(conditions, choices)
            
            hwynetwork = hwy_tcad.merge(
                hwy_load,
                left_on=["ID", "timePeriod"],
                right_on=["ID1", "timePeriod"],
                how="left")

            return hwynetwork

    def read_hwynetwork_raw(scenario_path)-> pd.DataFrame:
            # create list of loaded highway network files
            # and their respective ABM 5 time of day time periods
            time_periods = ['EA', 'AM', 'MD', 'PM', 'EV']

            # initialize loaded highway network DataFrame
            hwy_load = pd.DataFrame()

            # for each loaded highway network file
            for tp in time_periods:
                # read the loaded highway network
                data = pd.read_csv(
                    os.path.join(scenario_path, 'report','hwyload_' + tp + '.csv'),
                    usecols=["ID1",  # hwycov id
                            "AB_Time",  # A-B direction loaded time
                            "BA_Time",  # B-A direction loaded time
                            "AB_Flow",  # A-B directional flow
                            "BA_Flow"   # B-A directional flow
                            ]
                        )
                # fill NaN values with 0s
                data.fillna(value=0, inplace=True)
                data['timePeriod'] = tp

                # add to result DataFrame
                hwy_load = hwy_load.append(data)

            hwy_tcad = pd.read_csv(
                os.path.join(scenario_path, 'report','hwyTcad.csv'),
                usecols=["ID",  # hwycov id
                        "Length",  # length of link segment (miles)
                        "FC",  # link functional class
                        "ABTM_EA",  # A-B direction Early AM free flow time (minutes)
                        "ABTM_AM",  # A-B direction AM Peak free flow time (minutes)
                        "ABTM_MD",  # A-B direction Mid-day free flow time (minutes)
                        "ABTM_PM",  # A-B direction PM Peak free flow time (minutes)
                        "ABTM_EV",  # A-B direction Evening free flow time (minutes)
                        "BATM_EA",  # B-A direction Early AM free flow time (minutes)
                        "BATM_AM",  # B-A direction AM Peak free flow time (minutes)
                        "BATM_MD",  # B-A direction Mid-day free flow time (minutes)
                        "BATM_PM",  # B-A direction PM Peak free flow time (minutes)
                        "BATM_EV",  # B-A direction Evening free flow time (minutes)
                        "ABTX_EA",  # A-B direction Early AM intersection delay time (minutes)
                        "ABTX_AM",  # A-B direction AM Peak intersection delay time (minutes)
                        "ABTX_MD",  # A-B direction Mid-day intersection delay time (minutes)
                        "ABTX_PM",  # A-B direction PM Peak intersection delay time (minutes)
                        "ABTX_EV",  # A-B direction Evening intersection delay time (minutes)
                        "BATX_EA",  # B-A direction Early AM intersection delay time (minutes)
                        "BATX_AM",  # B-A direction AM Peak intersection delay time (minutes)
                        "BATX_MD",  # B-A direction Mid-day intersection delay time (minutes)
                        "BATX_PM",  # B-A direction PM Peak intersection delay time (minutes)
                        "BATX_EV",  # B-A direction Evening intersection delay time (minutes)
                        "ABPRELOAD_EA",  # A-B direction Early AM bus volume
                        "ABPRELOAD_AM",  # A-B direction AM Peak bus volume
                        "ABPRELOAD_MD",  # A-B direction Mid-day bus volume
                        "ABPRELOAD_PM",  # A-B direction PM Peak bus volume
                        "ABPRELOAD_EV",  # A-B direction Evening bus volume
                        "BAPRELOAD_EA",  # B-A direction Early AM bus volume
                        "BAPRELOAD_AM",  # B-A direction AM Peak bus volume
                        "BAPRELOAD_MD",  # B-A direction Mid-day bus volume
                        "BAPRELOAD_PM",  # B-A direction PM Peak bus volume
                        "BAPRELOAD_EV"  # B-A direction Evening bus volume]                        
                ]
            )
            # restructure file to long-format by ABM 5 time of day period
            hwy_tcad = pd.wide_to_long(
                df=hwy_tcad,
                stubnames=["ABTM",
                           "BATM",
                           "ABTX",
                           "BATX",
                           "ABPRELOAD",
                           "BAPRELOAD"],
                i=["ID",
                "Length",
                "FC"],
                j="timePeriod",
                sep="_",
                suffix="\w+").reset_index()
            
            # fill NaN values with 0s
            hwy_tcad.fillna(value=0, inplace=True)
            # add description of functional class to DataFrame
            conditions = [(hwy_tcad["FC"] == 1),
                          (hwy_tcad["FC"] == 2),
                          (hwy_tcad["FC"] == 3),
                          (hwy_tcad["FC"] == 4),
                          (hwy_tcad["FC"] == 5),
                          (hwy_tcad["FC"] == 6),
                          (hwy_tcad["FC"] == 7),
                          (hwy_tcad["FC"] == 8),
                          (hwy_tcad["FC"] == 9),
                          (hwy_tcad["FC"] == 10)]

            choices = ["Freeway",
                       "Prime Arterial",
                       "Major Arterial",
                       "Collector",
                       "Local Collector",
                       "Rural Collector",
                       "Local Road",
                       "Freeway Connector Ramp",
                       "Local Ramp",
                       "Zone Connector"]

            hwy_tcad["fc_desc"] = np.select(conditions, choices)
            
            hwynetwork = hwy_tcad.merge(
                hwy_load,
                left_on=["ID", "timePeriod"],
                right_on=["ID1", "timePeriod"],
                how="left")

            return hwynetwork

    def network_metric(scenario_path, metric, group):
        data = HighwayNetwork.read_hwynetwork(scenario_path)
        if metric == 'vmt':
            data['Total_VMT'] = (data.AB_Flow + data.BA_Flow + data.ABPRELOAD + data.BAPRELOAD) * data.Length
            group_column = 'Total_VMT'
        elif metric == 'vht':
            data['Total_VHT'] = ((data.AB_Flow + data.ABPRELOAD) * data.AB_Time / 60.0) + ((data.BA_Flow + data.BAPRELOAD) * data.BA_Time / 60.0)
            group_column = 'Total_VHT'
        elif metric == 'vhd':
            for vehicle_type, flow_col in [('Auto', 'AB_Flow_Auto'), ('Truck', 'AB_Flow_Truck'), ('Bus', 'ABPRELOAD')]:
                data[f'{vehicle_type}_VHD'] = ((data[flow_col] * (data.AB_Time - data.ABTM - data.ABTX) / 60.0) +
                                            (data[flow_col.replace('AB', 'BA')] * (data.BA_Time - data.BATM - data.BATX) / 60.0))
            data[['Auto_VHD', 'Truck_VHD', 'Bus_VHD']] = data[['Auto_VHD', 'Truck_VHD', 'Bus_VHD']].clip(lower=0)
            data['Total_VHD'] = data[['Auto_VHD', 'Truck_VHD', 'Bus_VHD']].sum(axis=1)
            group_column = 'Total_VHD'

        if group == 'timePeriod':
            grouped_data = data.groupby('timePeriod', as_index=False)[group_column].sum()
            total = grouped_data.sum(numeric_only=True)
            total['timePeriod'] = 'Total'
        else:
            grouped_data = data.groupby(['fc_desc'], as_index=False)[group_column].sum()
            total = grouped_data.sum(numeric_only=True)
            total['fc_desc'] = 'Total'

        return grouped_data.append(total, ignore_index=True)

    def network_metric_raw(scenario_path, metric, group):
        data = HighwayNetwork.read_hwynetwork(scenario_path)
        if metric == 'vmt':
            data['Total_VMT'] = (data.AB_Flow + data.BA_Flow) * data.Length
            group_column = 'Total_VMT'
        elif metric == 'vht':
            data['Total_VHT'] = (data.AB_Flow * data.AB_Time / 60.0) + (data.BA_Flow * data.BA_Time / 60.0)
            group_column = 'Total_VHT'
        elif metric == 'vhd':
            data['Total_VHD'] = ((data.AB_Flow * (data.AB_Time - data.ABTM - data.ABTX) / 60.0) +
                                            (data.BA_Flow * (data.BA_Time - data.BATM - data.BATX) / 60.0))
            group_column = 'Total_VHD'

        if group == 'timePeriod':
            grouped_data = data.groupby('timePeriod', as_index=False)[group_column].sum()
            total = grouped_data.sum(numeric_only=True)
            total['timePeriod'] = 'Total'
        else:
            grouped_data = data.groupby(['fc_desc'], as_index=False)[group_column].sum()
            total = grouped_data.sum(numeric_only=True)
            total['fc_desc'] = 'Total'

        return grouped_data.append(total, ignore_index=True)

class Skim:
    # compare function
    def skimReader(scenario_path, skim = ['transit','traffic'], time_periods = ['EA', 'AM', 'MD', 'PM', 'EV']):
        tp = time_periods

        if skim == 'transit':
             matrix = 'transit_skims_' + tp + '.omx'
             # desired sub matrices
             sub_matrices = [
                'WALK_PRM_TOTALIVTT__' + tp
                ,'WALK_PRM_LRTIVTT__'  + tp
                ,'WALK_PRM_CMRIVTT__'  + tp
                ,'WALK_PRM_EXPIVTT__'  + tp
                ,'WALK_PRM_BRTIVTT__'  + tp
                ,'WALK_LOC_TOTALIVTT__' + tp
                ,'WALK_MIX_TOTALIVTT__' + tp
                ,'WALK_PRM_XFERS__' + tp
                ,'WALK_LOC_XFERS__' + tp
                ,'WALK_MIX_XFERS__' + tp
                ]
        if skim == 'traffic':
             matrix = 'traffic_skims_' + tp + '.omx'
             sub_matrices = [
                'SOV_NT_M_DIST__' + tp
                #,'SOV_NT_M_TIME__' + tp
                #,'HOV2_M_HOVDIST__' + tp
                #,'HOV2_M_TIME__' + tp
                #,'HOV3_M_HOVDIST__' + tp
                #,'HOV3_M_TIME__' + tp
                ]

        inSkim = os.path.join(scenario_path, 'output', 'skims', matrix)        


        # read skim
        skims = omx.open_file(inSkim)
        zones = list(skims.mapping('zone_number').keys())

        # empty df with all zone-to-zone combinations
        zoneToZone = pd.DataFrame(list(itertools.product(list(range(1,4947+1,1)), repeat=2)))
        zoneToZone.columns = ['Origin', 'Destination']

        # read sub-matrices, convert from wide to long, and merge
        for sub_matrix in sub_matrices:
            od = pd.DataFrame(
                np.array(skims[sub_matrix]),
                zones,
                zones,
                )
            od = od.stack().reset_index().set_axis('Origin Destination {}'.format(sub_matrix).split(), axis=1)
            zoneToZone = zoneToZone.merge(od, on=['Origin', 'Destination'])
        
        return(zoneToZone)

    def regression_scatter_plot(matrix, OpenPath, EMME437, time_periods, skim):
        scenario_EMME437 = Skim.skimReader(EMME437, time_periods=time_periods,skim=skim)
        scenario_OpenPath = Skim.skimReader(OpenPath, time_periods=time_periods,skim=skim)
        
        EMME437_values = scenario_EMME437[matrix].values.reshape(-1, 1)
        OpenPath_values = scenario_OpenPath[matrix]

        # Fit Linear Regression model
        lin_reg = LinearRegression()
        lin_reg.fit(EMME437_values, OpenPath_values)
        y_pred = lin_reg.predict(EMME437_values)

        # Compute R² score and Root Mean Squared Error (RMSE)
        r2 = r2_score(OpenPath_values, y_pred)
        rmse = np.sqrt(mean_squared_error(OpenPath_values, y_pred))

        # Output regression parameters
        intercept = lin_reg.intercept_
        slope = lin_reg.coef_[0]

        print(f"Intercept: {intercept:.4f}")
        print(f"Slope: {slope:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")

        plt.figure(figsize=(9, 7))
        plt.scatter(EMME437_values, OpenPath_values, color='blue', alpha=0.6, label="Data Points")
        plt.plot(EMME437_values, y_pred, color='red', label=f"Regression Line\n$R^2$={r2:.4f},RMSE={rmse:.4f}")

        # Labels and title
        plt.xlabel("EMME437 Values")
        plt.ylabel("OpenPath Values")
        plt.title("Skim Comparison " + matrix)
        plt.legend()
        plt.grid(True)

        # Show plot
        plt.show()   


    def comparison(OpenPath, EMME437,time_periods, skim):
        # generate comparison
        scenario_OpenPath = Skim.skimReader(OpenPath, skim, time_periods)
        scenario_EMME437 = Skim.skimReader(EMME437, skim, time_periods)

        # OpenPaths minus EMME4.3.7 (exclude Origin and Destination columns)
        comparison = scenario_OpenPath.set_index(['Origin', 'Destination']) - scenario_EMME437.set_index(['Origin','Destination'])
        comparison.reset_index(inplace=True)

        return comparison

    def histogram_plot(matrix, OpenPath, EMME437,time_periods, skim):
        comparison = Skim.comparison(OpenPath, EMME437,time_periods, skim)

        # Set figure size
        plt.figure(figsize=(9, 7))  # Width=9, Height=7
        # Create histogram with specified bins
        counts, bins, patches = plt.hist(comparison[matrix], bins=np.arange(-5, 5.25, 0.25), edgecolor='black')

        # Set title and labels
        plt.title('Histogram of ' + matrix)
        plt.xlabel('Impedance difference (OpenPath - EMME437)')
        plt.ylabel('O-D pairs')

        # Change the y-axis range
        plt.ylim(0, 30000000) 

        # Add values on top of each bin
        for count, bin in zip(counts, bins):
        # Calculate the position for the text (center of each bin)
            plt.text(bin + 0.125, count, str(int(count)), ha='center', va='bottom', rotation=90)

        # Show the plot
        plt.show()