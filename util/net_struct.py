import glob


def get_feature_map(dataset):
    feature_file = open(f'./data/{dataset}/list.txt', 'r')
    feature_list = []
    for ft in feature_file:
        feature_list.append(ft.strip())

    return feature_list
# graph is 'fully-connect'
def get_fc_graph_struc(dataset):
    top_row = ['AF16_Temperature', 'AF16_Humidity', 'AF16_RSSI', 'AF16_SNR', 'AF16_TimeDifference', \
               'AF19_Temperature', 'AF19_Humidity', 'AF19_RSSI', 'AF19_SNR', 'AF19_TimeDifference',
               'AF22_Temperature', 'AF22_Humidity', 'AF22_RSSI', 'AF22_SNR', 'AF22_TimeDifference',
               'AF25_Temperature', 'AF25_Humidity', 'AF25_RSSI', 'AF25_SNR', 'AF25_TimeDifference',
               'AF28_Temperature', 'AF28_Humidity', 'AF28_RSSI', 'AF28_SNR', 'AF28_TimeDifference',
               'AF34_Temperature', 'AF34_Humidity', 'AF34_RSSI', 'AF34_SNR', 'AF34_TimeDifference', 
               'AF37_Temperature', 'AF37_Humidity', 'AF37_RSSI', 'AF37_SNR', 'AF37_TimeDifference',
               'AF40_Temperature', 'AF40_Humidity', 'AF40_RSSI', 'AF40_SNR', 'AF40_TimeDifference']

    middle_row = ['AF17_Temperature', 'AF17_Humidity', 'AF17_RSSI', 'AF17_SNR', 'AF17_TimeDifference', \
                  'AF20_Temperature', 'AF20_Humidity', 'AF20_RSSI', 'AF20_SNR', 'AF20_TimeDifference',
                  'AF23_Temperature', 'AF23_Humidity', 'AF23_RSSI', 'AF23_SNR', 'AF23_TimeDifference',
                  'AF26_Temperature', 'AF26_Humidity', 'AF26_RSSI', 'AF26_SNR', 'AF26_TimeDifference',
                  'AF29_Temperature', 'AF29_Humidity', 'AF29_RSSI', 'AF29_SNR', 'AF29_TimeDifference',
                  'AF35_Temperature', 'AF35_Humidity', 'AF35_RSSI', 'AF35_SNR', 'AF35_TimeDifference', 
                  'AF41_Temperature', 'AF41_Humidity', 'AF41_RSSI', 'AF41_SNR', 'AF41_TimeDifference']    

    bottom_row = ['AF18_Temperature', 'AF18_Humidity', 'AF18_RSSI', 'AF18_SNR', 'AF18_TimeDifference', \
                  'AF21_Temperature', 'AF21_Humidity', 'AF21_RSSI', 'AF21_SNR', 'AF21_TimeDifference',
                  'AF24_Temperature', 'AF24_Humidity', 'AF24_RSSI', 'AF24_SNR', 'AF24_TimeDifference',
                  'AF27_Temperature', 'AF27_Humidity', 'AF27_RSSI', 'AF27_SNR', 'AF27_TimeDifference',
                  'AF30_Temperature', 'AF30_Humidity', 'AF30_RSSI', 'AF30_SNR', 'AF30_TimeDifference',
                  'AF33_Temperature', 'AF33_Humidity', 'AF33_RSSI', 'AF33_SNR', 'AF33_TimeDifference',
                  'AF36_Temperature', 'AF36_Humidity', 'AF36_RSSI', 'AF36_SNR', 'AF36_TimeDifference', 
                  'AF42_Temperature', 'AF42_Humidity', 'AF42_RSSI', 'AF42_SNR', 'AF42_TimeDifference']    

    
    feature_file = open(f'./data/{dataset}/list.txt', 'r')

    struc_map = {}
    feature_list = []
    for ft in feature_file:
        feature_list.append(ft.strip())

    print(len(feature_list))

    # e.g. if ft is AF18_Temperature then struc_map[AF18_Temperature] = [AF18_Humidity, AF18_SNR, AF18_RSSI, AF18_Time_Difference]
    for ft in feature_list:
        if ft not in struc_map:
            struc_map[ft] = []
        for other_ft in feature_list:
            if other_ft is not ft:
                if other_ft[0:4] == ft[0:4]:
                    struc_map[ft].append(other_ft)

    '''
    for ft in feature_list:
        if ft not in struc_map:
            struc_map[ft] = []
        if ft in top_row:
            for other_ft in top_row:
                if other_ft is not ft:
                    struc_map[ft].append(other_ft)

        elif ft in middle_row:
            for other_ft in middle_row:
                if other_ft is not ft:
                    struc_map[ft].append(other_ft)

        elif ft in bottom_row:
            for other_ft in bottom_row:
                if other_ft is not ft:
                    struc_map[ft].append(other_ft)
    '''


    '''
    for ft in feature_list:
        if ft not in struc_map:
            struc_map[ft] = []

        for other_ft in feature_list:
            if other_ft is not ft:
                struc_map[ft].append(other_ft)
    '''

    return struc_map # struc map is a dictionary of ALL the sensors as keys and the values are ALL other sensors that are not the key (by ALL I mean sensors in both test.csv and train.csv)

def get_prior_graph_struc(dataset):
    feature_file = open(f'./data/{dataset}/features.txt', 'r')

    struc_map = {}
    feature_list = []
    for ft in feature_file:
        feature_list.append(ft.strip())

    for ft in feature_list:
        if ft not in struc_map:
            struc_map[ft] = []
        for other_ft in feature_list:
            if dataset == 'wadi' or dataset == 'wadi2':
                # same group, 1_xxx, 2A_xxx, 2_xxx
                if other_ft is not ft and other_ft[0] == ft[0]:
                    struc_map[ft].append(other_ft)
            elif dataset == 'swat':
                # FIT101, PV101
                if other_ft is not ft and other_ft[-3] == ft[-3]:
                    struc_map[ft].append(other_ft)

    
    return struc_map


if __name__ == '__main__':
    get_graph_struc()
 