import csv
from shapely.geometry import Point, mapping
from fiona import collection

def is_number(s):
    """ Returns True is string is a number. """
    try:
        float(s)
        return True
    except ValueError:
        return False

#csvfile = '../../data/field_calibration/Kiuic_AGB_2018_LiDAR.csv'
#csvfile = '../../data/field_calibration/Kiuic_400_live_trees.csv'
#csvfile = 'LiDAR_metrics_400_gliht_agb.csv'
csvfile = '../../data/lidar_calibration/Kiuic_400_live_biomass_unc.csv'
outfile = csvfile[:-4]+'.shp'
coord_x = 'x'
coord_y = 'y'

print(outfile)

properties_dtype = {}

with open(csvfile, 'rt') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        example = next(reader)
        for field in fields:
            if is_number(example[field]):
                properties_dtype[field]='float'
            else:
                properties_dtype[field]='str'

schema = { 'geometry': 'Point', 'properties': properties_dtype }
with collection(
    outfile, "w", "ESRI Shapefile", schema) as output:
    with open(csvfile, 'rt') as f:
        reader = csv.DictReader(f)
        for row in reader:
            point = Point(float(row[coord_x]), float(row[coord_y]))
            properties = {}
            for field in fields:
                properties[field]=row[field]
            output.write({
                'properties': properties,
                'geometry': mapping(point)
            })
