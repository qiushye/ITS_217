import sys
sys.path.append('.')
from population.population import population

if __name__ == '__main__':
    raw_dir = 'D:/启东数据/启东流量数据/'
    roads_path = data_dir + 'road_map.txt'
    interval = 30

    RN = road_network.roadmap(roads_path, data_dir + 'impute/')
    links = []
    for r in RN.roads:
        links += RN.road_info[r].A1


    