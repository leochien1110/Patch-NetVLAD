from VLADlib.VLAD import *
from VLADlib.Descriptors import *
import itertools
import argparse
import os
import cv2
import time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--query", required = True,
        help = "Path of a query image directory")
    ap.add_argument("-d", "--descriptor", required = True,
        help = "descriptors: SURF, SIFT or ORB")
    ap.add_argument("-dV", "--visualDictionary", required = True,
        help = "Path to the visual dictionary")
    ap.add_argument("-i", "--index", required = True,
        help = "Path of the Ball tree")

    args = vars(ap.parse_args())


    # args
    query_dir = args["query"]
    descriptorName=args["descriptor"]
    pathVD = args["visualDictionary"]
    treeIndex=args["index"]

    print('====>Loading treeIndex')
    with open(treeIndex, 'rb') as f:
        indexStructure=pickle.load(f)

    imageID=indexStructure[0]
    tree = indexStructure[1]
    pathImageData = indexStructure[2]
    print(pathImageData)

    print('====>Loading visual dictionary')
    with open(pathVD, 'rb') as f:
        visualDictionary=pickle.load(f)

    cv2.namedWindow("Query", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Match", cv2.WINDOW_AUTOSIZE)

    print('press any key to start')
    cv2.waitKey(0)

    start = time.time()
    for filename in sorted(os.listdir(query_dir)):
        print(filename)
        img_path = os.path.join(query_dir,filename)
        print(img_path)
        frame = cv2.imread(img_path)

        if frame is None:
            print('Could not load the image')
            continue

        # find top-k candidate: extract features -> VLAD -> search in tree
        dist,ind = query(img_path,1,descriptorName, visualDictionary,tree)

        # print(dist)
        # print(ind)

        ind=list(itertools.chain.from_iterable(ind))
        
        print(f"[Query/Matched] {filename} <--> {imageID[ind[0]]}")

        # display the query
        end = time.time()
        elapse = end - start
        start = end
        print(f"FPS: {1/elapse}")
        frame = cv2.resize(frame, (640, 480))
        cv2.putText(frame, f"FPS: {1/elapse}", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,200,255), 2)
        cv2.imshow("Query", frame)

        # load the result image and display it
        result = cv2.imread(imageID[ind[0]])
        result = cv2.resize(result, (640, 480))
        cv2.imshow("Match", result)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    print('Done!')
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()