from VLADlib.VLAD import *
from VLADlib.Descriptors import *
import argparse
import pickle
import glob
import cv2

#parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "Path to the directory that contains the images")
ap.add_argument("-n", "--descriptor", required = True,
	help = "descriptor = SURF, SIFT or  ORB")
ap.add_argument("-w", "--numberOfVisualWords", required = True,
	help = "number of visual words or clusters to be computed")
ap.add_argument("--visualDictionary_path", required = False, default="visualDictionary/",
	help = "Path to where the computed visualDictionary will be stored")
ap.add_argument("-l", "--leafSize", required = True,
	help = "Size of the leafs of the Ball tree")
ap.add_argument("-o", "--output", required = True,
	help = "Path to where VLAD descriptors will be stored")
args = vars(ap.parse_args())


#reading arguments
path = args["dataset"]
descriptorName=args["descriptor"]
k = int(args["numberOfVisualWords"])
vD_path = args["visualDictionary_path"]
leafSize = int(args["leafSize"])
output=args["output"]


# 1. computing the descriptors
dict={"SURF":describeSURF,"SIFT":describeSIFT,"ORB":describeORB}
descriptors=getDescriptors(path, dict[descriptorName])

# 2. training data by using k-means
visualDictionary=kMeansDictionary(descriptors,k)
# save visualDictionary to file
vDout = vD_path + "visualDictionary" + str(k) + descriptorName
file=vDout+".pickle"	
with open(file, 'wb') as f:
	pickle.dump(visualDictionary, f)
print("The visual dictionary  is saved in "+file)

# 3. computing the VLAD descriptors
dict={"SURF":describeSURF,"SIFT":describeSIFT,"ORB":describeORB}  
V, imageID = getVLADDescriptors(path,dict[descriptorName],visualDictionary)

# 4. index VLAD with ball-tree DS (can be replaced by K-D tree)
tree = indexBallTree(V,leafSize)

# 5. output - save VLAD to file for query matching
file=output+".pickle"

with open(file, 'wb') as f:
	pickle.dump([imageID,tree,path], f,pickle.HIGHEST_PROTOCOL)

print("The ball tree index is saved at "+file)